# Security Policy

## Supported Versions

We actively support the following versions with security updates:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

If you discover a security vulnerability in this project, please report it responsibly:

### How to Report

1. **DO NOT** create a public issue for security vulnerabilities
2. Email security concerns to: mukuljangra5@gmail.com
3. Include detailed information about the vulnerability
4. Provide steps to reproduce the issue
5. Include your contact information

### What to Include

- Description of the vulnerability
- Steps to reproduce
- Potential impact assessment
- Suggested fix (if available)
- Your contact information

### Response Timeline

- **Initial Response**: Within 48 hours
- **Status Update**: Within 7 days
- **Fix Timeline**: Varies by severity
  - Critical: 1-3 days
  - High: 1-2 weeks
  - Medium: 2-4 weeks
  - Low: Next release cycle

## Security Best Practices

### For Users

1. **Environment Variables**: Always store database credentials in environment variables, never hardcode them
2. **Network Security**: Run the server behind a firewall or VPN in production
3. **Database Permissions**: Use read-only database users when possible
4. **SSL/TLS**: Enable SSL connections to your MySQL server
5. **Access Control**: Restrict access to the MCP server to authorized users only
6. **Regular Updates**: Keep dependencies and MySQL server updated

### For Developers

1. **Input Validation**: All SQL queries are validated before execution
2. **Parameterized Queries**: Use parameterized queries to prevent SQL injection
3. **Error Handling**: Avoid exposing sensitive information in error messages
4. **Logging**: Be careful not to log sensitive information like passwords
5. **Code Review**: All changes undergo security review

## Security Features

- **SQL Injection Protection**: Parameterized queries and input validation
- **Read-Only Mode**: Option to run in read-only mode for safety
- **Connection Security**: SSL/TLS support for database connections
- **Access Logging**: Comprehensive logging of database operations
- **Privilege Auditing**: Built-in tools to audit database privileges
- **Error Sanitization**: Error messages are sanitized to prevent information disclosure

## Vulnerability Disclosure Policy

We are committed to working with security researchers and the community to verify and address security vulnerabilities. We ask that you:

1. Give us reasonable time to address issues before public disclosure
2. Avoid accessing, modifying, or deleting data during testing
3. Only test against systems you own or have permission to test
4. Respect user privacy and comply with applicable laws

## Security Updates

Security updates are released as patch versions (e.g., 1.0.1, 1.0.2) and are clearly marked in the release notes. Subscribe to repository notifications to stay informed about security updates.

## Attribution

We appreciate the security research community and will acknowledge researchers who responsibly disclose vulnerabilities (unless they prefer to remain anonymous).

## Contact

For security-related questions or concerns:
- Email: mukuljangra5@gmail.com
- GitHub: Create a private vulnerability report
- Encrypted Communication: Available upon request

---

**Note**: This security policy is subject to change. Please check back regularly for updates.
