"""SQL Server client for executing stored procedures."""

import logging

try:
    import pymssql
except ImportError:
    pymssql = None

logger = logging.getLogger()


class SQLServerClient:
    """Client for SQL Server database operations."""

    def __init__(
        self,
        server: str,
        database: str,
        user: str,
        password: str,
        port: int = 1433,
    ):
        """
        Initialize SQL Server client.

        Args:
            server: SQL Server host
            database: Database name
            user: Username
            password: Password
            port: SQL Server port (default: 1433)
        """
        if pymssql is None:
            raise ImportError(
                "pymssql is not installed. Please install it to use SQL Server integration."
            )

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
        """Context manager entry."""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
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

    def execute_sp_vectorizacion(self, id_carga: int, usumod: str = "n8n") -> None:
        """
        Execute SP_CARGA_CONOC_PROCESO_VECTORIZACION stored procedure.

        This should be called after embeddings are generated but before Weaviate upsert.

        Args:
            id_carga: Load ID
            usumod: User who modified (default: 'n8n')
        """
        if not self.connection:
            raise RuntimeError("Not connected to SQL Server")

        try:
            logger.info(f"Executing SP_CARGA_CONOC_PROCESO_VECTORIZACION with ID_CARGA={id_carga}, USUMOD={usumod}")

            cursor = self.connection.cursor()
            cursor.execute(
                "EXEC SP_CARGA_CONOC_PROCESO_VECTORIZACION @ID_CARGA=%d, @USUMOD=%s",
                (id_carga, usumod)
            )
            self.connection.commit()
            cursor.close()

            logger.info("Successfully executed SP_CARGA_CONOC_PROCESO_VECTORIZACION")

        except Exception as e:
            logger.error(f"Error executing SP_CARGA_CONOC_PROCESO_VECTORIZACION: {e}")
            raise

    def execute_sp_completado(self, id_carga: int, usumod: str = "n8n") -> None:
        """
        Execute SP_CARGA_CONOC_PROCESO_COMPLETADO stored procedure.

        This should be called after Weaviate upsert is complete.

        Args:
            id_carga: Load ID
            usumod: User who modified (default: 'n8n')
        """
        if not self.connection:
            raise RuntimeError("Not connected to SQL Server")

        try:
            logger.info(f"Executing SP_CARGA_CONOC_PROCESO_COMPLETADO with ID_CARGA={id_carga}, USUMOD={usumod}")

            cursor = self.connection.cursor()
            cursor.execute(
                "EXEC SP_CARGA_CONOC_PROCESO_COMPLETADO @ID_CARGA=%d, @USUMOD=%s",
                (id_carga, usumod)
            )
            self.connection.commit()
            cursor.close()

            logger.info("Successfully executed SP_CARGA_CONOC_PROCESO_COMPLETADO")

        except Exception as e:
            logger.error(f"Error executing SP_CARGA_CONOC_PROCESO_COMPLETADO: {e}")
            raise
