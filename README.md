# MySQL MCP Server

A comprehensive Model Context Protocol (MCP) server for MySQL databases, designed to empower AI assistants and LLMs with advanced database access, management, and diagnostic capabilities.

## Repository

- **GitHub**: [https://github.com/mukul975/mcp-mysql-server](https://github.com/mukul975/mcp-mysql-server)
- **Author**: mukul975
- **License**: MIT

## Core Features

- **Execute SQL Queries**: Support for safe execution of `SELECT`, `INSERT`, `UPDATE`, `DELETE`, and other commands with read-only validation.
- **Database Introspection**: List tables, view schemas, fetch sample data, and explore indexes and constraints.
- **Advanced Diagnostics**: Comprehensive analysis tools covering fragmentation, slow queries, deadlocks, buffer pool usage, partitioning, and query performance.
- **Security and User Management**: Tools for auditing user privileges, managing users and roles, SSL/TLS configuration audit, and monitoring audit logs.
- **Backup and Replication Monitoring**: Backup health status, binary log and replication status, replication lag monitoring, and recovery readiness.
- **Server and System Monitoring**: View system variables, process lists, query cache stats, memory usage breakdowns, and plugin/component status.
- **Performance Insights**: Adaptive index and query optimizer analyses, event scheduler management, and resource consumption evaluations.

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

Or install key dependencies separately:

```bash
pip install "mcp[cli]"
pip install mysql-connector-python
```

## Configuration

Configure via environment variables:

| Variable         | Description            | Default   |
| ---------------- | ---------------------- | --------- |
| `MYSQL_HOST`     | MySQL server host      | `localhost` |
| `MYSQL_PORT`     | MySQL server port      | `3306`    |
| `MYSQL_USER`     | MySQL username         | `root`    |
| `MYSQL_PASSWORD` | MySQL password         | (empty)   |
| `MYSQL_DATABASE` | Default database name  | (empty)   |

Example (Linux/macOS):

```bash
export MYSQL_HOST=localhost
export MYSQL_PORT=3306
export MYSQL_USER=myuser
export MYSQL_PASSWORD=mypassword
export MYSQL_DATABASE=mydatabase
```

Example (Windows PowerShell):

```powershell
$env:MYSQL_HOST = "localhost"
$env:MYSQL_PORT = "3306"
$env:MYSQL_USER = "myuser"
$env:MYSQL_PASSWORD = "mypassword"
$env:MYSQL_DATABASE = "mydatabase"
```

## Usage

Run the server:

- Default stdio transport:

```bash
python mysql_server.py
```

- SSE transport (for web clients):

```bash
python mysql_server.py --transport sse
```

Get help:

```bash
python mysql_server.py --help
```

## Integration

To integrate with Claude Desktop, update your config file (`%APPDATA%/Claude/claude_desktop_config.json` on Windows or `~/Library/Application Support/Claude/claude_desktop_config.json` on macOS):

```json
{
  "mcpServers": {
    "mysql": {
      "command": "python",
      "args": ["path/to/mysql_server.py"],
      "env": {
        "MYSQL_HOST": "localhost",
        "MYSQL_PORT": "3306",
        "MYSQL_USER": "your_username",
        "MYSQL_PASSWORD": "your_password",
        "MYSQL_DATABASE": "your_database"
      }
    }
  }
}
```

## Example Workflows

- List all tables: use `list_tables` tool or access `mysql://tables` resource.
- Inspect table schema: use `describe_table` tool or `mysql://schema/{table_name}`.
- Execute queries: use `execute_sql` for select or data modification queries.
- Analyze slow queries and deadlocks.
- Audit user privileges and monitor SSL/TLS connections.
- Monitor replication lag and binary logs for health.

## Included Tools

The server includes a rich set of tools such as:

- mysql_query, list_mysql_tables, mysql_table_schema, mysql_table_data
- mysql_table_indexes, mysql_table_size, mysql_table_status
- mysql_fragmentation_analysis, mysql_index_optimization_suggestions, mysql_slow_query_analysis
- mysql_deadlock_detection, mysql_buffer_pool_cache_diagnostics
- mysql_user_privileges, mysql_create_user, mysql_drop_user, mysql_change_user_password
- mysql_backup_health_check, mysql_replication_lag_monitoring, mysql_ssl_tls_configuration_audit
- mysql_server_health_dashboard, mysql_performance_recommendations
- mysql_event_scheduler, mysql_partition_management_recommendations
- And many more diagnostic, operational, and security tools.

## Security Considerations

- Use secure connections when possible.
- Store credentials in environment variables.
- Only use the server in trusted environments or behind network security.
- All queries are validated for safety, but always review for injection risks.
- The server includes comprehensive privilege auditing tools.

## Error Handling

- Robust error reporting for connection, syntax, permission, and network errors.
- Structured error response format for easy automated handling.

## Development

Project structure:

```
mcp-mysql-server/
├── mysql_server.py      # Core server code with tools and protocols
├── requirements.txt     # Python dependencies
├── README.md            # This documentation
└── pyproject.toml       # Optional project metadata
```

### Testing

- Ensure MySQL server running and reachable.
- Configure environment variables.
- Run `python mysql_server.py`
- Optionally test with `mcp dev mysql_server.py`

## Contributing

- Fork repository
- Create branches for features or fixes
- Add tests and documentation
- Submit pull requests for review

## License

MIT License — Open source and free to use
