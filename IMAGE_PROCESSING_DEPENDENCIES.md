# 🖼️ IMAGE PROCESSING DEPENDENCIES - DRHP SYSTEM

## Required NPM Packages

Add the following dependencies to your `package.json`:

```json
{
  "dependencies": {
    "sharp": "^0.32.6",
    "tesseract.js": "^4.1.4"
  }
}
```

## Installation Commands

```bash
# Install image processing dependencies
npm install sharp tesseract.js

# Or using yarn
yarn add sharp tesseract.js
```

## Package Details

### Sharp (^0.32.6)
- **Purpose**: High-performance image processing
- **Features**: 
  - Image resizing and optimization
  - Format conversion (JPEG, PNG, WebP, TIFF, etc.)
  - Image enhancement (sharpening, normalization)
  - Metadata extraction
- **Platform Support**: Cross-platform with native bindings
- **Performance**: Fastest Node.js image processing library

### Tesseract.js (^4.1.4)
- **Purpose**: OCR (Optical Character Recognition)
- **Features**:
  - Text extraction from images
  - Multi-language support (100+ languages)
  - Confidence scoring
  - Word and line-level recognition
  - Bounding box detection
- **Languages Supported**: English, Hindi, Tamil, Gujarati, Bengali, Telugu, Marathi, Kannada
- **Browser/Node**: Works in both browser and Node.js environments

## System Requirements

### For Sharp:
- **Node.js**: 14.15.0 or higher
- **Memory**: Minimum 512MB available RAM
- **Disk Space**: ~50MB for installation
- **Platform**: Windows, macOS, Linux

### For Tesseract.js:
- **Node.js**: 12.0.0 or higher
- **Memory**: Minimum 1GB available RAM (OCR processing)
- **Disk Space**: ~100MB for language data
- **Network**: Internet connection for downloading language models

## Environment Variables

Add to your `.env` file:

```env
# Image Processing Configuration
IMAGE_PROCESSING_ENABLED=true
OCR_LANGUAGES=eng+hin+tam+guj+ben+tel+mar+kan
MAX_IMAGE_SIZE=50MB
IMAGE_QUALITY_THRESHOLD=medium
OCR_CONFIDENCE_THRESHOLD=60

# Tesseract Configuration
TESSERACT_CACHE_PATH=./cache/tesseract
TESSERACT_WORKER_POOL_SIZE=2
```

## Usage Examples

### Basic Image Processing
```javascript
const sharp = require('sharp');

// Enhance image for OCR
const enhancedImage = await sharp(inputBuffer)
  .resize(2000, 2000, { fit: 'inside' })
  .normalize()
  .sharpen()
  .greyscale()
  .png({ quality: 100 })
  .toBuffer();
```

### OCR Text Extraction
```javascript
const { createWorker } = require('tesseract.js');

// Initialize OCR worker
const worker = await createWorker('eng');
await worker.loadLanguage('eng+hin');
await worker.initialize('eng+hin');

// Extract text
const { data } = await worker.recognize(imageBuffer);
console.log('Extracted text:', data.text);
```

## Performance Optimization

### Sharp Optimization:
- Use appropriate image formats (PNG for text, JPEG for photos)
- Resize images before processing to reduce memory usage
- Enable progressive JPEG for web delivery
- Use WebP format for better compression

### Tesseract Optimization:
- Preprocess images (enhance contrast, remove noise)
- Use appropriate page segmentation modes
- Limit OCR to specific regions of interest
- Cache OCR workers to avoid initialization overhead

## Troubleshooting

### Common Issues:

1. **Sharp Installation Fails**
   ```bash
   # Clear npm cache and reinstall
   npm cache clean --force
   npm install sharp --platform=win32 --arch=x64
   ```

2. **Tesseract Language Download Fails**
   ```bash
   # Manual language data download
   mkdir -p ./node_modules/tesseract.js-core/lang-data
   # Download language files manually
   ```

3. **Memory Issues with Large Images**
   ```javascript
   // Process images in chunks or reduce resolution
   const resized = await sharp(largeImage)
     .resize(1000, 1000, { fit: 'inside' })
     .toBuffer();
   ```

4. **OCR Accuracy Issues**
   ```javascript
   // Enhance image preprocessing
   const enhanced = await sharp(image)
     .normalize()
     .sharpen()
     .threshold(128) // Binary threshold
     .toBuffer();
   ```

## Production Considerations

### Docker Support:
```dockerfile
# Add to Dockerfile
RUN apt-get update && apt-get install -y \
    libvips-dev \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-hin

# Install Node.js dependencies
COPY package*.json ./
RUN npm ci --only=production
```

### Memory Management:
- Monitor memory usage during image processing
- Implement image processing queues for high-volume scenarios
- Use streaming for large file processing
- Set appropriate Node.js memory limits

### Security:
- Validate image file types and sizes
- Sanitize extracted text content
- Implement rate limiting for image processing endpoints
- Use secure temporary file handling

## Integration Status

✅ **Fully Integrated** with DRHP Generation System
✅ **ASI-Powered** analysis and content extraction
✅ **Multi-format Support** for all common image types
✅ **Production Ready** with error handling and fallbacks
✅ **Scalable Architecture** for enterprise deployment

## Next Steps

1. Install the required dependencies
2. Configure environment variables
3. Test image processing with sample files
4. Deploy with appropriate resource allocation
5. Monitor performance and optimize as needed
