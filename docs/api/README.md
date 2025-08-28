# IPE API Documentation

## Overview

The Interactionist Phylogeny Engine (IPE) provides a comprehensive REST API for running evolutionary simulations, managing physiological state spaces, and exporting results. The API is built using FastAPI and provides automatic OpenAPI/Swagger documentation.

## Base URL

- **Local Development**: `http://localhost:8000`
- **Staging**: `https://staging.ipe.velottalab.com`
- **Production**: `https://ipe.velottalab.com`

## Interactive Documentation

The API provides automatically generated interactive documentation:

- **Swagger UI**: `/docs`
- **ReDoc**: `/redoc`
- **OpenAPI Schema**: `/openapi.json`

## API Version

Current API version: `v1`

All endpoints are prefixed with `/api/v1/`

## Response Format

All API responses follow a consistent JSON format:

```json
{
  "data": {},
  "status": "success|error",
  "message": "Human readable message",
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Error Handling

The API uses standard HTTP status codes:

- **200**: Success
- **201**: Created
- **400**: Bad Request
- **401**: Unauthorized
- **403**: Forbidden
- **404**: Not Found
- **422**: Validation Error
- **500**: Internal Server Error

Error responses include detailed information:

```json
{
  "detail": "Error description",
  "status_code": 400,
  "timestamp": "2024-01-01T00:00:00Z"
}
```

## Rate Limiting

API endpoints are rate-limited to ensure fair usage:

- **Anonymous**: 100 requests/hour
- **Authenticated**: 1000 requests/hour
- **Premium**: 10000 requests/hour

Rate limit headers are included in responses:

```
X-RateLimit-Limit: 1000
X-RateLimit-Remaining: 999
X-RateLimit-Reset: 1609459200
```

## Content Types

The API supports the following content types:

- **Request**: `application/json`
- **Response**: `application/json`
- **File Uploads**: `multipart/form-data`
- **File Downloads**: `application/octet-stream`, `text/csv`, `image/png`, etc.

## Quick Start

1. **Create a simulation**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/simulations" \
     -H "Content-Type: application/json" \
     -d '{"name": "Test Simulation", "duration": 100}'
   ```

2. **List simulations**:
   ```bash
   curl "http://localhost:8000/api/v1/simulations"
   ```

3. **Get simulation state space**:
   ```bash
   curl "http://localhost:8000/api/v1/states/{simulation_id}"
   ```

4. **Export results**:
   ```bash
   curl -X POST "http://localhost:8000/api/v1/export/simulation/{simulation_id}" \
     -H "Content-Type: application/json" \
     -d '{"format": "csv", "data_types": ["simulation_data"]}'
   ```

## Next Steps

- [Endpoint Reference](endpoints.md) - Detailed endpoint documentation
- [Code Examples](examples.md) - Integration examples in different languages
- [Authentication Guide](#authentication) - Setup and usage
- [Rate Limits](#rate-limiting) - Usage policies

## Support

For API support and questions:
- **Documentation**: [docs.ipe.velottalab.com](https://docs.ipe.velottalab.com)
- **Issues**: [GitHub Issues](https://github.com/mnechromancer/RIPE/issues)
- **Email**: support@ipe.velottalab.com