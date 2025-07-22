# Contributing to MCP MySQL Server

Thank you for your interest in contributing to the MCP MySQL Server project! We welcome contributions from the community and are grateful for your help in making this project better.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Getting Started](#getting-started)
- [How to Contribute](#how-to-contribute)
- [Development Setup](#development-setup)
- [Coding Standards](#coding-standards)
- [Testing](#testing)
- [Submitting Changes](#submitting-changes)
- [Reporting Issues](#reporting-issues)
- [Feature Requests](#feature-requests)
- [Community](#community)

## Code of Conduct

By participating in this project, you are expected to uphold our Code of Conduct. Please be respectful, inclusive, and constructive in all interactions.

### Our Pledge

- Use welcoming and inclusive language
- Be respectful of differing viewpoints and experiences
- Gracefully accept constructive criticism
- Focus on what is best for the community
- Show empathy towards other community members

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/your-username/mcp-mysql-server.git
   cd mcp-mysql-server
   ```
3. **Set up the development environment** (see Development Setup below)
4. **Create a feature branch** for your contribution

## How to Contribute

### Types of Contributions

- **Bug fixes**: Fix issues or unexpected behavior
- **New features**: Add new MySQL tools or MCP functionality
- **Documentation**: Improve README, add examples, write guides
- **Performance improvements**: Optimize queries or server performance
- **Security enhancements**: Improve security features or practices
- **Testing**: Add or improve test coverage
- **Code quality**: Refactor code, improve error handling

### Contribution Process

1. Check existing issues and pull requests to avoid duplicates
2. Create an issue to discuss major changes before implementing
3. Follow the development setup and coding standards
4. Write tests for new functionality
5. Update documentation as needed
6. Submit a pull request with a clear description

## Development Setup

### Prerequisites

- Python 3.8 or higher
- MySQL 5.7 or 8.x server (for testing)
- Git

### Local Setup

1. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements-dev.txt  # If available
   ```

3. **Set up test database:**
   ```bash
   # Create a test database
   mysql -u root -p -e "CREATE DATABASE mcp_test;"
   ```

4. **Configure environment variables:**
   ```bash
   export MYSQL_HOST=localhost
   export MYSQL_PORT=3306
   export MYSQL_USER=root
   export MYSQL_PASSWORD=your_password
   export MYSQL_DATABASE=mcp_test
   ```

5. **Run the server:**
   ```bash
   python mysql_server.py
   ```

## Coding Standards

### Python Style Guide

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guidelines
- Use type hints where appropriate
- Write docstrings for all functions and classes
- Keep functions focused and small
- Use meaningful variable and function names

### Code Formatting

We recommend using:
- **Black** for code formatting: `black mysql_server.py`
- **isort** for import sorting: `isort mysql_server.py`
- **flake8** for linting: `flake8 mysql_server.py`

### Documentation Style

- Use clear, concise language
- Include examples in docstrings
- Document all parameters and return values
- Update README.md for new features

### Example Function Documentation

```python
def mysql_analyze_table(table_name: str, schema_name: str = "public") -> Dict[str, Any]:
    """
    Analyze a MySQL table to update key distribution statistics.
    
    This function runs MySQL's ANALYZE TABLE command to update the key
    distribution statistics used by the query optimizer.
    
    Args:
        table_name: Name of the table to analyze
        schema_name: Database schema name (default: "public")
        
    Returns:
        Dict containing analysis results and status information
        
    Raises:
        MySQLError: If the table doesn't exist or analysis fails
        ConnectionError: If database connection is lost
        
    Example:
        >>> result = mysql_analyze_table("users")
        >>> print(result["status"])
        "OK"
    """
```

## Testing

### Running Tests

```bash
# Run all tests
python -m pytest

# Run with coverage
python -m pytest --cov=mysql_server

# Run specific test file
python -m pytest tests/test_mysql_tools.py
```

### Writing Tests

- Write tests for all new functionality
- Use descriptive test names
- Test both success and failure cases
- Mock external dependencies (database connections)
- Include edge cases and boundary conditions

### Test Structure

```python
def test_mysql_analyze_table_success():
    """Test successful table analysis."""
    # Setup
    # Execute
    # Assert

def test_mysql_analyze_table_invalid_table():
    """Test analysis with non-existent table."""
    # Test error handling
```

## Submitting Changes

### Pull Request Process

1. **Create a feature branch:**
   ```bash
   git checkout -b feature/add-new-mysql-tool
   ```

2. **Make your changes:**
   - Follow coding standards
   - Add tests
   - Update documentation

3. **Commit your changes:**
   ```bash
   git add .
   git commit -m "Add new MySQL performance analysis tool"
   ```

4. **Push to your fork:**
   ```bash
   git push origin feature/add-new-mysql-tool
   ```

5. **Create a pull request** on GitHub

### Pull Request Guidelines

- **Title**: Use a clear, descriptive title
- **Description**: Explain what changes were made and why
- **Testing**: Describe how the changes were tested
- **Documentation**: Note any documentation updates
- **Breaking Changes**: Clearly mark any breaking changes

### Pull Request Template

```markdown
## Description
Brief description of changes made.

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Documentation update
- [ ] Performance improvement
- [ ] Security enhancement

## Testing
- [ ] Tests pass locally
- [ ] New tests added for new functionality
- [ ] Manual testing completed

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] No breaking changes (or clearly documented)
```

## Reporting Issues

### Bug Reports

Use the bug report template and include:

- **Environment**: OS, Python version, MySQL version
- **Steps to reproduce**: Detailed steps
- **Expected behavior**: What should happen
- **Actual behavior**: What actually happens
- **Error messages**: Full error output
- **Logs**: Relevant log entries

### Security Issues

**DO NOT** create public issues for security vulnerabilities. Please see our [Security Policy](SECURITY.md) for responsible disclosure procedures.

## Feature Requests

Before requesting a feature:

1. Check if it already exists or is planned
2. Search existing issues and discussions
3. Consider if it fits the project's scope

Include in your request:
- **Use case**: Why is this needed?
- **Proposed solution**: How might it work?
- **Alternatives**: Other options considered
- **Additional context**: Screenshots, examples, etc.

## Community

### Communication Channels

- **GitHub Issues**: For bugs and feature requests
- **GitHub Discussions**: For general questions and ideas
- **Pull Requests**: For code contributions

### Getting Help

- Check the [README.md](README.md) for basic usage
- Search existing issues for similar problems
- Create a new issue with the "question" label

### Recognition

Contributors are acknowledged in:
- Repository contributors list
- Release notes for significant contributions
- Special mentions for major features or fixes

## Development Roadmap

### Current Priorities

1. Enhanced security features
2. Performance optimization
3. Additional MySQL 8.x features
4. Improved error handling
5. Comprehensive test coverage

### Future Goals

1. Support for other database systems
2. Web-based management interface
3. Advanced monitoring and alerting
4. Integration with popular DevOps tools

## Resources

- [MySQL Documentation](https://dev.mysql.com/doc/)
- [Model Context Protocol Specification](https://spec.modelcontextprotocol.io/)
- [Python Database API Specification](https://www.python.org/dev/peps/pep-0249/)

---

Thank you for contributing to MCP MySQL Server! Your efforts help make database management more accessible to AI systems and developers worldwide.
