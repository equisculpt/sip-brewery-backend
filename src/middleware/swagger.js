/**
 * 📚 SWAGGER/OPENAPI DOCUMENTATION MIDDLEWARE
 * 
 * Integrates OpenAPI documentation with Swagger UI
 * Provides interactive API documentation and testing
 */

const swaggerUi = require('swagger-ui-express');
const YAML = require('yamljs');
const path = require('path');
const fs = require('fs');
const logger = require('../utils/logger');

class SwaggerIntegration {
  constructor() {
    this.swaggerDocument = null;
    this.setupSwagger();
  }

  setupSwagger() {
    try {
      const swaggerPath = path.join(__dirname, '../../docs/openapi.yaml');
      
      if (fs.existsSync(swaggerPath)) {
        this.swaggerDocument = YAML.load(swaggerPath);
        logger.info('✅ OpenAPI documentation loaded successfully');
      } else {
        logger.warn('⚠️ OpenAPI documentation file not found');
        this.createFallbackDoc();
      }
    } catch (error) {
      logger.error('❌ Failed to load OpenAPI documentation:', error);
      this.createFallbackDoc();
    }
  }

  createFallbackDoc() {
    this.swaggerDocument = {
      openapi: '3.0.3',
      info: {
        title: 'SIP Brewery API',
        version: '1.0.0',
        description: 'Universe-Class Financial ASI Platform API'
      },
      servers: [
        {
          url: process.env.NODE_ENV === 'production' 
            ? 'https://api.sipbrewery.com' 
            : 'http://localhost:3000',
          description: process.env.NODE_ENV === 'production' ? 'Production' : 'Development'
        }
      ],
      paths: {
        '/health': {
          get: {
            summary: 'Health check',
            responses: {
              '200': {
                description: 'System is healthy'
              }
            }
          }
        }
      }
    };
  }

  getSwaggerOptions() {
    return {
      explorer: true,
      swaggerOptions: {
        docExpansion: 'none',
        filter: true,
        showRequestHeaders: true,
        showCommonExtensions: true,
        tryItOutEnabled: true
      },
      customCss: `
        .swagger-ui .topbar { display: none; }
        .swagger-ui .info .title { color: #1f2937; font-size: 2rem; }
        .swagger-ui .info .description { font-size: 1rem; }
        .swagger-ui .scheme-container { background: #f8fafc; padding: 1rem; border-radius: 0.5rem; }
        .swagger-ui .opblock.opblock-get .opblock-summary-method { background: #10b981; }
        .swagger-ui .opblock.opblock-post .opblock-summary-method { background: #3b82f6; }
        .swagger-ui .opblock.opblock-put .opblock-summary-method { background: #f59e0b; }
        .swagger-ui .opblock.opblock-delete .opblock-summary-method { background: #ef4444; }
      `,
      customSiteTitle: 'SIP Brewery API Documentation',
      customfavIcon: '/favicon.ico'
    };
  }

  setupRoutes(app) {
    // Swagger JSON endpoint
    app.get('/api-docs.json', (req, res) => {
      res.setHeader('Content-Type', 'application/json');
      res.send(this.swaggerDocument);
    });

    // Swagger UI
    app.use('/api-docs', 
      swaggerUi.serve, 
      swaggerUi.setup(this.swaggerDocument, this.getSwaggerOptions())
    );

    // Alternative documentation routes
    app.use('/docs', 
      swaggerUi.serve, 
      swaggerUi.setup(this.swaggerDocument, this.getSwaggerOptions())
    );

    // API specification download
    app.get('/openapi.yaml', (req, res) => {
      const yamlPath = path.join(__dirname, '../../docs/openapi.yaml');
      if (fs.existsSync(yamlPath)) {
        res.setHeader('Content-Type', 'text/yaml');
        res.sendFile(yamlPath);
      } else {
        res.status(404).json({ error: 'OpenAPI specification not found' });
      }
    });

    logger.info('📚 Swagger documentation available at:');
    logger.info('   - /api-docs (Swagger UI)');
    logger.info('   - /docs (Alternative)');
    logger.info('   - /api-docs.json (JSON spec)');
    logger.info('   - /openapi.yaml (YAML spec)');
  }

  // Middleware to add OpenAPI validation (optional)
  validateRequest() {
    return (req, res, next) => {
      // TODO: Add request validation against OpenAPI schema
      // This can be implemented using libraries like express-openapi-validator
      next();
    };
  }

  // Generate API documentation from routes (auto-discovery)
  generateDocsFromRoutes(app) {
    const routes = [];
    
    // Extract routes from Express app
    app._router.stack.forEach((middleware) => {
      if (middleware.route) {
        routes.push({
          path: middleware.route.path,
          methods: Object.keys(middleware.route.methods)
        });
      }
    });

    logger.info(`📊 Discovered ${routes.length} API routes`);
    return routes;
  }

  // Health check for documentation
  getDocumentationHealth() {
    return {
      status: this.swaggerDocument ? 'healthy' : 'degraded',
      documentationAvailable: !!this.swaggerDocument,
      endpoints: {
        swaggerUI: '/api-docs',
        jsonSpec: '/api-docs.json',
        yamlSpec: '/openapi.yaml'
      },
      lastUpdated: new Date().toISOString()
    };
  }
}

// Export singleton instance
const swaggerIntegration = new SwaggerIntegration();

module.exports = {
  SwaggerIntegration,
  swaggerIntegration,
  setupSwagger: (app) => swaggerIntegration.setupRoutes(app),
  validateRequest: () => swaggerIntegration.validateRequest(),
  getDocumentationHealth: () => swaggerIntegration.getDocumentationHealth()
};
