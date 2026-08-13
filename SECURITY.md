# Security Policy

## Supported Versions

We release patches for security vulnerabilities for the following versions:

| Version | Supported          |
| ------- | ------------------ |
| 0.0.x   | :white_check_mark: |

## Reporting a Vulnerability

We take the security of Sanityops Agent seriously. If you have discovered a security vulnerability, we appreciate your help in disclosing it to us in a responsible manner.

### How to Report

Please report security vulnerabilities by emailing:

**security@sanityops.org**

Do not create a public GitHub issue for security vulnerabilities.

### What to Include

When reporting, please include:

1. **Description** of the vulnerability
2. **Steps to reproduce** the issue
3. **Potential impact** and attack scenarios
4. **Suggested fix** (if you have one)
5. **Your contact information** for follow-up

### Response Timeline

| Stage | Timeline |
|-------|----------|
| Acknowledgment | Within 48 hours |
| Initial Assessment | Within 7 days |
| Fix Development | Varies by complexity |
| Release | As soon as possible after fix |

### Disclosure Policy

- We follow a **coordinated disclosure** approach
- We will work with you to understand and address the issue
- Public disclosure will happen **after a fix is released**
- We will credit you in the release notes if you wish

### Security Best Practices

When using Sanityops Agent:

- **Never commit API keys** to version control
- Use environment variables or secure secret management
- Keep your dependencies up to date
- Review the permissions of tools you enable

### Scope

The following are **in scope** for security reports:

- Vulnerabilities in the core library
- Security issues in the authentication/authorization flows
- Sensitive data exposure
- Injection vulnerabilities

The following are **out of scope**:

- Social engineering attacks
- Physical attacks
- Attacks requiring compromised credentials
- DoS attacks without significant resource consumption

Thank you for helping keep Sanityops Agent and our users safe!