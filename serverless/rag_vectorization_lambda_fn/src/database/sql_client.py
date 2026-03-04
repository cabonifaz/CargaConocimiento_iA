"""SQL Server client for RAG ingestion stored procedures."""

import logging

try:
    import pymssql
except ImportError:
    pymssql = None

logger = logging.getLogger()

WORKER_NAME = "vectorization-lambda"


class SQLServerClient:
    """Client for SQL Server operations during the vectorization stage."""

    def __init__(
        self,
        server: str,
        database: str,
        user: str,
        password: str,
        port: int = 1433,
    ):
        if pymssql is None:
            raise ImportError("pymssql is not installed.")

        if not all([server, database, user, password]):
            raise ValueError(
                "Database credentials are incomplete. Required: server, database, user, password"
            )

        self.server = server
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.connection = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def connect(self) -> None:
        """Establish connection to SQL Server."""
        try:
            logger.info(f"Connecting to SQL Server: {self.server}:{self.port}/{self.database}")
            self.connection = pymssql.connect(
                server=self.server,
                user=self.user,
                password=self.password,
                database=self.database,
                port=self.port,
                timeout=30,
                login_timeout=10,
            )
            logger.info("Successfully connected to SQL Server")
        except Exception as e:
            logger.error(f"Failed to connect to SQL Server: {e}")
            raise

    def close(self) -> None:
        """Close SQL Server connection."""
        if self.connection:
            try:
                self.connection.close()
                logger.info("SQL Server connection closed")
            except Exception as e:
                logger.warning(f"Error closing SQL Server connection: {e}")

    def iniciar_etapa(
        self,
        id_proceso: int,
        id_etapa: int,
        estado_procesando: int,
        usucre: str = WORKER_NAME,
    ) -> int:
        """
        Execute SP_RAG_INGESTA_INICIAR_ETAPA.

        Sets the process to state `estado_procesando` and creates a log entry.

        Args:
            id_proceso: Process ID (from RAG_INGESTA_PROCESOS)
            id_etapa: Stage type — 3 = Vectorización
            estado_procesando: Processing state — 6 (Vectorizando)
            usucre: Worker identifier

        Returns:
            ID_LOG of the newly created log entry.
        """
        if not self.connection:
            raise RuntimeError("Not connected to SQL Server")

        try:
            logger.info(
                f"SP_RAG_INGESTA_INICIAR_ETAPA id_proceso={id_proceso}, "
                f"etapa={id_etapa}, estado={estado_procesando}"
            )
            cursor = self.connection.cursor(as_dict=True)
            cursor.execute(
                "EXEC SP_RAG_INGESTA_INICIAR_ETAPA "
                "@ID_PROCESO=%d, @ID_ETAPA_INGESTA_RAG=%d, "
                "@ESTADO_PROCESANDO=%d, @USUCRE=%s",
                (id_proceso, id_etapa, estado_procesando, usucre),
            )

            # First result set: message — check for errors
            first_row = cursor.fetchone()
            if not first_row or first_row.get("ID_TIPO_MENSAJE") != 2:
                msg = first_row.get("MENSAJE") if first_row else "no response from SP"
                raise RuntimeError(f"SP_RAG_INGESTA_INICIAR_ETAPA failed: {msg}")

            # Second result set: ID_LOG
            cursor.nextset()
            log_row = cursor.fetchone()
            if not log_row or "ID_LOG" not in log_row:
                raise RuntimeError("SP_RAG_INGESTA_INICIAR_ETAPA did not return ID_LOG")

            self.connection.commit()
            id_log = log_row["ID_LOG"]
            logger.info(f"Etapa iniciada — ID_LOG={id_log}")
            return id_log

        except Exception as e:
            logger.error(f"Error in iniciar_etapa: {e}")
            raise

    def completar_etapa(
        self,
        id_log: int,
        id_proceso: int,
        estado_siguiente: int,
        ruta_resultado: str,
        costo_usd: float,
        usumod: str = WORKER_NAME,
    ) -> None:
        """
        Execute SP_RAG_INGESTA_COMPLETAR_ETAPA.

        Marks the log entry as successful and advances the process to
        `estado_siguiente`. For vectorization, ruta_resultado is None
        (vectors live in Weaviate, not S3).

        Args:
            id_log: Log entry ID (from iniciar_etapa)
            id_proceso: Process ID
            estado_siguiente: 7 = Cargado (terminal success)
            ruta_resultado: None for this stage
            costo_usd: Estimated Bedrock embedding cost
            usumod: Worker identifier
        """
        if not self.connection:
            raise RuntimeError("Not connected to SQL Server")

        try:
            logger.info(
                f"SP_RAG_INGESTA_COMPLETAR_ETAPA id_log={id_log}, "
                f"id_proceso={id_proceso}, estado_siguiente={estado_siguiente}, "
                f"costo_usd={costo_usd:.6f}"
            )
            cursor = self.connection.cursor(as_dict=True)
            cursor.execute(
                "EXEC SP_RAG_INGESTA_COMPLETAR_ETAPA "
                "@ID_LOG=%d, @ID_PROCESO=%d, @ESTADO_SIGUIENTE=%d, "
                "@RUTA_RESULTADO=%s, @COSTO_USD=%s, @USUMOD=%s",
                (id_log, id_proceso, estado_siguiente, ruta_resultado, f"{costo_usd:.6f}", usumod),
            )
            self.connection.commit()
            logger.info(f"Etapa completada — proceso {id_proceso} → estado {estado_siguiente}")

        except Exception as e:
            logger.error(f"Error in completar_etapa: {e}")
            raise

    def get_model_costs(self) -> dict:
        """
        Query PARAMETROS for model costs (ID_MAESTRO=14, DESCRIPCION='COSTOS_MODELOS_X_MILLON_TKN').

        Returns:
            {id_modelo: (cost_input_per_million, cost_output_per_million)}
            cost_output_per_million is 0.0 for models with no separate output pricing.
        """
        if not self.connection:
            raise RuntimeError("Not connected to SQL Server")

        try:
            cursor = self.connection.cursor()
            cursor.execute(
                "SELECT NUM2, CAST(STRING1 AS FLOAT), CAST(STRING2 AS FLOAT) "
                "FROM PARAMETROS "
                "WHERE ID_MAESTRO = 14 "
                "  AND DESCRIPCION = 'COSTOS_MODELOS_X_MILLON_TKN' "
                "  AND ID_ESTADO_REGISTRO = 1"
            )
            rows = cursor.fetchall()
            costs = {
                int(row[0]): (float(row[1] or 0), float(row[2] or 0))
                for row in rows
            }
            logger.info(f"Loaded model costs from DB: {costs}")
            return costs

        except Exception as e:
            logger.error(f"Error in get_model_costs: {e}")
            raise

    def fallar_etapa(
        self,
        id_log: int,
        id_proceso: int,
        mensaje_error: str,
        usumod: str = WORKER_NAME,
    ) -> None:
        """
        Execute SP_RAG_INGESTA_FALLAR_ETAPA.

        Marks the log entry as failed and sets the process state to 8 (Error).

        Args:
            id_log: Log entry ID (from iniciar_etapa)
            id_proceso: Process ID
            mensaje_error: Error description (truncated to 200 chars)
            usumod: Worker identifier
        """
        if not self.connection:
            raise RuntimeError("Not connected to SQL Server")

        try:
            mensaje_error = mensaje_error[:200]
            logger.info(
                f"SP_RAG_INGESTA_FALLAR_ETAPA id_log={id_log}, "
                f"id_proceso={id_proceso}, error='{mensaje_error}'"
            )
            cursor = self.connection.cursor(as_dict=True)
            cursor.execute(
                "EXEC SP_RAG_INGESTA_FALLAR_ETAPA "
                "@ID_LOG=%d, @ID_PROCESO=%d, @MENSAJE_ERROR=%s, @USUMOD=%s",
                (id_log, id_proceso, mensaje_error, usumod),
            )
            self.connection.commit()
            logger.info(f"Proceso {id_proceso} marcado como Error (estado 8)")

        except Exception as e:
            logger.error(f"Error in fallar_etapa: {e}")
            raise
