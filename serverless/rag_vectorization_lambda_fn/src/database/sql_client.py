"""SQL Server client for RAG ingestion stored procedures."""

import logging
from typing import Optional

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
        logger.info("Connecting to SQL Server: %s:%s/%s", self.server, self.port, self.database)
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

    def close(self) -> None:
        """Close SQL Server connection."""
        if self.connection:
            try:
                self.connection.close()
                logger.info("SQL Server connection closed")
            except Exception as e:
                logger.warning("Error closing SQL Server connection: %s", e)

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

        logger.info(
            "SP_RAG_INGESTA_INICIAR_ETAPA id_proceso=%s, etapa=%s, estado=%s",
            id_proceso, id_etapa, estado_procesando,
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
        logger.info("Etapa iniciada — ID_LOG=%s", id_log)
        return id_log

    def completar_etapa(
        self,
        id_log: int,
        id_proceso: int,
        estado_siguiente: int,
        ruta_resultado: str,
        costo_usd: float,
        usumod: str = WORKER_NAME,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        id_modelo: Optional[int] = None,
        avanzar_estado: bool = True,
    ) -> None:
        """
        Execute SP_RAG_INGESTA_COMPLETAR_ETAPA.

        Marks the log entry as successful. When avanzar_estado=True (default)
        also advances the process to `estado_siguiente`.

        Args:
            id_log: Log entry ID (from iniciar_etapa or crear_log)
            id_proceso: Process ID
            estado_siguiente: 7 = Cargado (terminal success)
            ruta_resultado: None for vectorization stage
            costo_usd: Bedrock cost (from actual token counts × DB rates)
            usumod: Worker identifier
            input_tokens: Input token count for this model call
            output_tokens: Output token count for this model call (None for embedding)
            id_modelo: Model ID (maps to PARAMETROS ID_MAESTRO=14)
            avanzar_estado: False = update log only, do NOT advance process state
        """
        if not self.connection:
            raise RuntimeError("Not connected to SQL Server")

        logger.info(
            "SP_RAG_INGESTA_COMPLETAR_ETAPA id_log=%s, id_proceso=%s, "
            "estado_siguiente=%s, costo_usd=%.6f, avanzar=%s",
            id_log, id_proceso, estado_siguiente, costo_usd, avanzar_estado,
        )
        cursor = self.connection.cursor(as_dict=True)
        cursor.execute(
            "EXEC SP_RAG_INGESTA_COMPLETAR_ETAPA "
            "@ID_LOG=%d, @ID_PROCESO=%d, @ESTADO_SIGUIENTE=%d, "
            "@RUTA_RESULTADO=%s, @COSTO_USD=%s, @USUMOD=%s, "
            "@INPUT_TOKENS=%s, @OUTPUT_TOKENS=%s, @ID_MODELO=%s, @AVANZAR_ESTADO=%d",
            (
                id_log, id_proceso, estado_siguiente,
                ruta_resultado, f"{costo_usd:.6f}", usumod,
                input_tokens, output_tokens, id_modelo,
                1 if avanzar_estado else 0,
            ),
        )
        self.connection.commit()
        logger.info("Etapa completada — proceso %s → estado %s (avanzar=%s)", id_proceso, estado_siguiente, avanzar_estado)

    def crear_log(
        self,
        id_proceso: int,
        id_etapa: int,
        usucre: str = WORKER_NAME,
    ) -> int:
        """
        Execute SP_RAG_INGESTA_CREAR_LOG.

        Inserts a new log row for an additional sub-log within the same etapa
        (e.g. second model in the vectorization stage) without touching process state.

        Args:
            id_proceso: Process ID
            id_etapa: Stage type — 3 = Vectorización
            usucre: Worker identifier

        Returns:
            ID_LOG of the newly created log entry.
        """
        if not self.connection:
            raise RuntimeError("Not connected to SQL Server")

        logger.info(
            "SP_RAG_INGESTA_CREAR_LOG id_proceso=%s, id_etapa=%s",
            id_proceso, id_etapa,
        )
        cursor = self.connection.cursor(as_dict=True)
        cursor.execute(
            "EXEC SP_RAG_INGESTA_CREAR_LOG @ID_PROCESO=%d, @ID_ETAPA=%d, @USUCRE=%s",
            (id_proceso, id_etapa, usucre),
        )

        # First result set: ID_LOG
        log_row = cursor.fetchone()
        if not log_row or "ID_LOG" not in log_row:
            raise RuntimeError("SP_RAG_INGESTA_CREAR_LOG did not return ID_LOG")

        self.connection.commit()
        id_log = int(log_row["ID_LOG"])
        logger.info("Log adicional creado — ID_LOG=%s", id_log)
        return id_log

    def get_model_costs(self) -> dict:
        """
        Query PARAMETROS for model costs (ID_MAESTRO=14, DESCRIPCION='COSTOS_MODELOS_X_MILLON_TKN').

        Returns:
            {id_modelo: (cost_input_per_million, cost_output_per_million)}
            cost_output_per_million is 0.0 for models with no separate output pricing.
        """
        if not self.connection:
            raise RuntimeError("Not connected to SQL Server")

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
        logger.info("Loaded model costs from DB: %s", costs)
        return costs

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

        mensaje_error = mensaje_error[:200]
        logger.info(
            "SP_RAG_INGESTA_FALLAR_ETAPA id_log=%s, id_proceso=%s, error='%s'",
            id_log, id_proceso, mensaje_error,
        )
        cursor = self.connection.cursor(as_dict=True)
        cursor.execute(
            "EXEC SP_RAG_INGESTA_FALLAR_ETAPA "
            "@ID_LOG=%d, @ID_PROCESO=%d, @MENSAJE_ERROR=%s, @USUMOD=%s",
            (id_log, id_proceso, mensaje_error, usumod),
        )
        self.connection.commit()
        logger.info("Proceso %s marcado como Error (estado 8)", id_proceso)
