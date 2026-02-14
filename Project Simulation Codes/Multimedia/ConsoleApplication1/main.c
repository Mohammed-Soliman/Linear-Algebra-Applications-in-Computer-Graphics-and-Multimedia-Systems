#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <math.h>
#include <errno.h>

#pragma pack(push,1)
typedef struct {
    uint16_t bfType;      // 'BM' = 0x4D42
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
} BITMAPFILEHEADER;
#pragma pack(pop)

#pragma pack(push,1)
typedef struct {
    uint32_t biSize;          // size of this header (40)
    int32_t  biWidth;
    int32_t  biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    int32_t  biXPelsPerMeter;
    int32_t  biYPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
} BITMAPINFOHEADER;
#pragma pack(pop)

// Apply color transformation matrix to BMP image
void applyColorTransform(uint8_t* image, int width, int height, float transform[3][3]) {
    if (!image || width <= 0 || height <= 0) return;

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t index = ((size_t)y * (size_t)width + (size_t)x) * 3;

            float b = (float)image[index];
            float g = (float)image[index + 1];
            float r = (float)image[index + 2];

            float r_new = transform[0][0] * r + transform[0][1] * g + transform[0][2] * b;
            float g_new = transform[1][0] * r + transform[1][1] * g + transform[1][2] * b;
            float b_new = transform[2][0] * r + transform[2][1] * g + transform[2][2] * b;

            image[index] = (uint8_t)fmaxf(0.0f, fminf(255.0f, b_new));
            image[index + 1] = (uint8_t)fmaxf(0.0f, fminf(255.0f, g_new));
            image[index + 2] = (uint8_t)fmaxf(0.0f, fminf(255.0f, r_new));
        }
    }
}

// Grayscale transformation matrix
float grayscaleMatrix[3][3] = {
    {0.212655f, 0.715158f, 0.072187f},
    {0.212655f, 0.715158f, 0.072187f},
    {0.212655f, 0.715158f, 0.072187f}
};

// Color inversion using matrix multiplication
void applyColorInversion(uint8_t* image, int width, int height) {
    if (!image || width <= 0 || height <= 0) return;

    // Inversion matrix: multiply by -1 and add 255
    // This represents the transformation: out = -1 * in + 255
    float inversionMatrix[3][3] = {
        {-1.0f,  0.0f,  0.0f},  // R' = -1*R + 0*G + 0*B
        { 0.0f, -1.0f,  0.0f},  // G' = 0*R + -1*G + 0*B
        { 0.0f,  0.0f, -1.0f}   // B' = 0*R + 0*G + -1*B
    };

    // Translation vector (added after multiplication)
    float translation[3] = { 255.0f, 255.0f, 255.0f };

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            size_t index = ((size_t)y * (size_t)width + (size_t)x) * 3;

            float b = (float)image[index];
            float g = (float)image[index + 1];
            float r = (float)image[index + 2];

            // Apply matrix multiplication: output = matrix * input + translation
            float r_new = inversionMatrix[0][0] * r + inversionMatrix[0][1] * g + inversionMatrix[0][2] * b + translation[0];
            float g_new = inversionMatrix[1][0] * r + inversionMatrix[1][1] * g + inversionMatrix[1][2] * b + translation[1];
            float b_new = inversionMatrix[2][0] * r + inversionMatrix[2][1] * g + inversionMatrix[2][2] * b + translation[2];

            // Clamp values to [0, 255]
            image[index] = (uint8_t)fmax(0, fmin(255, b_new));
            image[index + 1] = (uint8_t)fmax(0, fmin(255, g_new));
            image[index + 2] = (uint8_t)fmax(0, fmin(255, r_new));
        }
    }
}

// Sepia transformation matrix
float sepiaMatrix[3][3] = {
    {0.3588f, 0.7044f, 0.1368f},
    {0.2990f, 0.5870f, 0.1140f},
    {0.2392f, 0.4696f, 0.0912f}
};

// Image downscaling by factor of 2 (4 pixels -> 1 pixel average)
uint8_t* downscaleImage(uint8_t* image, int width, int height, int* new_width, int* new_height) {
    if (!image || width <= 1 || height <= 1) return NULL;

    *new_width = width / 2;
    *new_height = height / 2;

    size_t new_size = (size_t)(*new_width) * (size_t)(*new_height) * 3;
    uint8_t* new_image = (uint8_t*)malloc(new_size);
    if (!new_image) return NULL;

    for (int y = 0; y < *new_height; y++) {
        for (int x = 0; x < *new_width; x++) {
            // Average of 4 pixels: (2x,2y), (2x+1,2y), (2x,2y+1), (2x+1,2y+1)
            size_t idx1 = ((size_t)(2 * y) * (size_t)width + (size_t)(2 * x)) * 3;
            size_t idx2 = ((size_t)(2 * y) * (size_t)width + (size_t)(2 * x + 1)) * 3;
            size_t idx3 = ((size_t)(2 * y + 1) * (size_t)width + (size_t)(2 * x)) * 3;
            size_t idx4 = ((size_t)(2 * y + 1) * (size_t)width + (size_t)(2 * x + 1)) * 3;

            size_t new_idx = ((size_t)y * (size_t)(*new_width) + (size_t)x) * 3;

            for (int c = 0; c < 3; c++) {
                int sum = image[idx1 + c] + image[idx2 + c] + image[idx3 + c] + image[idx4 + c];
                new_image[new_idx + c] = (uint8_t)(sum / 4);
            }
        }
    }

    return new_image;
}

// Image upscaling by factor of 2 (nearest neighbor)
uint8_t* upscaleImage(uint8_t* image, int width, int height, int* new_width, int* new_height) {
    if (!image || width <= 0 || height <= 0) return NULL;

    *new_width = width * 2;
    *new_height = height * 2;

    size_t new_size = (size_t)(*new_width) * (size_t)(*new_height) * 3;
    uint8_t* new_image = (uint8_t*)malloc(new_size);
    if (!new_image) return NULL;

    for (int y = 0; y < *new_height; y++) {
        for (int x = 0; x < *new_width; x++) {
            // Nearest neighbor: map back to original pixel
            int src_x = x / 2;
            int src_y = y / 2;

            size_t src_idx = ((size_t)src_y * (size_t)width + (size_t)src_x) * 3;
            size_t dst_idx = ((size_t)y * (size_t)(*new_width) + (size_t)x) * 3;

            new_image[dst_idx] = image[src_idx];
            new_image[dst_idx + 1] = image[src_idx + 1];
            new_image[dst_idx + 2] = image[src_idx + 2];
        }
    }

    return new_image;
}

// Image convolution operation (3x3 kernel)
void applyConvolution(uint8_t* image, int width, int height, float kernel[3][3], float factor) {
    if (!image || width <= 2 || height <= 2 || !kernel) return;

    size_t bytes = (size_t)width * (size_t)height * 3;
    uint8_t* temp = (uint8_t*)malloc(bytes);
    if (!temp) return;

    memcpy(temp, image, bytes);

    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            float sum_r = 0.0f, sum_g = 0.0f, sum_b = 0.0f;

            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    size_t idx = ((size_t)(y + ky) * (size_t)width + (size_t)(x + kx)) * 3;
                    float k = kernel[ky + 1][kx + 1];

                    sum_b += (float)temp[idx] * k;
                    sum_g += (float)temp[idx + 1] * k;
                    sum_r += (float)temp[idx + 2] * k;
                }
            }

            size_t idx = ((size_t)y * (size_t)width + (size_t)x) * 3;
            image[idx] = (uint8_t)fmaxf(0.0f, fminf(255.0f, sum_b * factor));
            image[idx + 1] = (uint8_t)fmaxf(0.0f, fminf(255.0f, sum_g * factor));
            image[idx + 2] = (uint8_t)fmaxf(0.0f, fminf(255.0f, sum_r * factor));
        }
    }
    free(temp);
}


// ==================== DCT COMPRESSION FUNCTIONS ====================

// DCT coefficient function
float dct_c(int u) {
    return (u == 0) ? (1.0f / sqrtf(2.0f)) : 1.0f;
}

// Apply 8x8 DCT to a single channel
void applyDCT8x8(float block[8][8]) {
    float temp[8][8] = { 0 };
    float output[8][8] = { 0 };

    const float pi = 3.14159265358979323846f;

    // First pass: DCT on rows
    for (int y = 0; y < 8; y++) {
        for (int u = 0; u < 8; u++) {
            float sum = 0.0f;
            for (int x = 0; x < 8; x++) {
                float angle = ((2.0f * x + 1.0f) * u * pi) / 16.0f;
                sum += block[y][x] * cosf(angle);
            }
            temp[y][u] = 0.5f * dct_c(u) * sum;
        }
    }

    // Second pass: DCT on columns
    for (int u = 0; u < 8; u++) {
        for (int v = 0; v < 8; v++) {
            float sum = 0.0f;
            for (int y = 0; y < 8; y++) {
                float angle = ((2.0f * y + 1.0f) * v * pi) / 16.0f;
                sum += temp[y][u] * cosf(angle);
            }
            output[u][v] = 0.5f * dct_c(v) * sum;
        }
    }

    // Copy result back
    memcpy(block, output, sizeof(output));
}

// Apply inverse DCT to a single channel
void applyIDCT8x8(float block[8][8]) {
    float temp[8][8] = { 0 };
    float output[8][8] = { 0 };

    const float pi = 3.14159265358979323846f;

    // First pass: IDCT on rows
    for (int y = 0; y < 8; y++) {
        for (int x = 0; x < 8; x++) {
            float sum = 0.0f;
            for (int u = 0; u < 8; u++) {
                float angle = ((2.0f * x + 1.0f) * u * pi) / 16.0f;
                sum += dct_c(u) * block[y][u] * cosf(angle);
            }
            temp[y][x] = 0.5f * sum;
        }
    }

    // Second pass: IDCT on columns
    for (int x = 0; x < 8; x++) {
        for (int y = 0; y < 8; y++) {
            float sum = 0.0f;
            for (int v = 0; v < 8; v++) {
                float angle = ((2.0f * y + 1.0f) * v * pi) / 16.0f;
                sum += dct_c(v) * temp[v][x] * cosf(angle);
            }
            output[y][x] = 0.5f * sum;
        }
    }

    // Copy result back
    memcpy(block, output, sizeof(output));
}

// Apply DCT compression to image (simplified JPEG-like)
uint8_t* applyDCTCompression(uint8_t* image, int width, int height, int* new_width, int* new_height, float quality) {
    if (!image || width < 8 || height < 8) return NULL;

    // Ensure dimensions are multiples of 8
    *new_width = (width / 8) * 8;
    *new_height = (height / 8) * 8;

    size_t new_size = (size_t)(*new_width) * (size_t)(*new_height) * 3;
    uint8_t* new_image = (uint8_t*)malloc(new_size);
    if (!new_image) return NULL;

    memcpy(new_image, image, new_size);

    // Process each 8x8 block
    for (int block_y = 0; block_y < *new_height; block_y += 8) {
        for (int block_x = 0; block_x < *new_width; block_x += 8) {
            // Process each color channel
            for (int channel = 0; channel < 3; channel++) {
                float block[8][8] = { 0 };

                // Extract block
                for (int y = 0; y < 8; y++) {
                    for (int x = 0; x < 8; x++) {
                        size_t idx = ((size_t)(block_y + y) * (size_t)(*new_width) + (size_t)(block_x + x)) * 3;
                        block[y][x] = (float)new_image[idx + channel];
                    }
                }

                // Apply DCT
                applyDCT8x8(block);

                // Simple quantization (simulate compression)
                for (int y = 0; y < 8; y++) {
                    for (int x = 0; x < 8; x++) {
                        // Zero out high frequencies based on quality
                        if ((x + y) > (8 * (1.0f - quality))) {
                            block[y][x] = 0.0f;
                        }
                    }
                }

                // Apply inverse DCT
                applyIDCT8x8(block);

                // Write back block
                for (int y = 0; y < 8; y++) {
                    for (int x = 0; x < 8; x++) {
                        size_t idx = ((size_t)(block_y + y) * (size_t)(*new_width) + (size_t)(block_x + x)) * 3;
                        float value = fmaxf(0.0f, fminf(255.0f, block[y][x]));
                        new_image[idx + channel] = (uint8_t)value;
                    }
                }
            }
        }
    }

    return new_image;
}

// ==================== BLOCK COMPRESSION (BC1/DXT1) FUNCTIONS ====================

typedef struct {
    uint8_t r, g, b;
} RGBColor;

// Calculate color distance
float colorDistance(RGBColor c1, RGBColor c2) {
    return sqrtf((c1.r - c2.r) * (c1.r - c2.r) +
        (c1.g - c2.g) * (c1.g - c2.g) +
        (c1.b - c2.b) * (c1.b - c2.b));
}

// BC1/DXT1 block compression for 4x4 block
void compressBC1Block(uint8_t* block_pixels, uint8_t* compressed) {
    RGBColor colors[16];
    RGBColor min_color = { 255, 255, 255 };
    RGBColor max_color = { 0, 0, 0 };

    // Find min and max colors in 4x4 block
    for (int i = 0; i < 16; i++) {
        colors[i].r = block_pixels[i * 3 + 2];
        colors[i].g = block_pixels[i * 3 + 1];
        colors[i].b = block_pixels[i * 3 + 0];

        if (colors[i].r < min_color.r) min_color.r = colors[i].r;
        if (colors[i].g < min_color.g) min_color.g = colors[i].g;
        if (colors[i].b < min_color.b) min_color.b = colors[i].b;

        if (colors[i].r > max_color.r) max_color.r = colors[i].r;
        if (colors[i].g > max_color.g) max_color.g = colors[i].g;
        if (colors[i].b > max_color.b) max_color.b = colors[i].b;
    }

    // Store endpoint colors (simplified - using min/max)
    compressed[0] = max_color.r;
    compressed[1] = max_color.g;
    compressed[2] = max_color.b;
    compressed[3] = min_color.r;
    compressed[4] = min_color.g;
    compressed[5] = min_color.b;

    // Calculate interpolated colors
    RGBColor palette[4];
    palette[0] = max_color;
    palette[3] = min_color;

    // Color 2: (2*max_color + min_color) / 3
    palette[1].r = (2 * max_color.r + min_color.r) / 3;
    palette[1].g = (2 * max_color.g + min_color.g) / 3;
    palette[1].b = (2 * max_color.b + min_color.b) / 3;

    // Color 3: (max_color + 2*min_color) / 3
    palette[2].r = (max_color.r + 2 * min_color.r) / 3;
    palette[2].g = (max_color.g + 2 * min_color.g) / 3;
    palette[2].b = (max_color.b + 2 * min_color.b) / 3;

    // Store indices (2 bits per pixel)
    for (int i = 0; i < 16; i++) {
        int best_index = 0;
        float best_distance = colorDistance(colors[i], palette[0]);

        for (int j = 1; j < 4; j++) {
            float dist = colorDistance(colors[i], palette[j]);
            if (dist < best_distance) {
                best_distance = dist;
                best_index = j;
            }
        }

        // Store index in compressed data (simplified storage)
        if (i < 8) {
            compressed[6] |= (best_index << (6 - i * 2));
        }
        else {
            compressed[7] |= (best_index << (14 - i * 2));
        }
    }
}

// Apply BC1/DXT1 compression to image
uint8_t* applyBC1Compression(uint8_t* image, int width, int height, int* new_width, int* new_height) {
    if (!image || width < 4 || height < 4) return NULL;

    // Ensure dimensions are multiples of 4
    *new_width = (width / 4) * 4;
    *new_height = (height / 4) * 4;

    size_t new_size = (size_t)(*new_width) * (size_t)(*new_height) * 3;
    uint8_t* new_image = (uint8_t*)malloc(new_size);
    if (!new_image) return NULL;

    // Process each 4x4 block
    for (int block_y = 0; block_y < *new_height; block_y += 4) {
        for (int block_x = 0; block_x < *new_width; block_x += 4) {
            // Extract 4x4 block
            uint8_t block_pixels[16 * 3] = { 0 };
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    size_t src_idx = ((size_t)(block_y + y) * (size_t)(*new_width) + (size_t)(block_x + x)) * 3;
                    size_t block_idx = (y * 4 + x) * 3;

                    block_pixels[block_idx] = image[src_idx];
                    block_pixels[block_idx + 1] = image[src_idx + 1];
                    block_pixels[block_idx + 2] = image[src_idx + 2];
                }
            }

            // Compress block
            uint8_t compressed[8] = { 0 };
            compressBC1Block(block_pixels, compressed);

            // Decompress for visualization (in real usage, you'd store compressed data)
            RGBColor palette[4];
            palette[0].r = compressed[0]; palette[0].g = compressed[1]; palette[0].b = compressed[2];
            palette[3].r = compressed[3]; palette[3].g = compressed[4]; palette[3].b = compressed[5];
            palette[1].r = (2 * palette[0].r + palette[3].r) / 3;
            palette[1].g = (2 * palette[0].g + palette[3].g) / 3;
            palette[1].b = (2 * palette[0].b + palette[3].b) / 3;
            palette[2].r = (palette[0].r + 2 * palette[3].r) / 3;
            palette[2].g = (palette[0].g + 2 * palette[3].g) / 3;
            palette[2].b = (palette[0].b + 2 * palette[3].b) / 3;

            // Write decompressed block
            for (int y = 0; y < 4; y++) {
                for (int x = 0; x < 4; x++) {
                    size_t dst_idx = ((size_t)(block_y + y) * (size_t)(*new_width) + (size_t)(block_x + x)) * 3;
                    int pixel_index = y * 4 + x;
                    int palette_index;

                    if (pixel_index < 8) {
                        palette_index = (compressed[6] >> (6 - pixel_index * 2)) & 0x03;
                    }
                    else {
                        palette_index = (compressed[7] >> (14 - pixel_index * 2)) & 0x03;
                    }

                    new_image[dst_idx] = palette[palette_index].b;
                    new_image[dst_idx + 1] = palette[palette_index].g;
                    new_image[dst_idx + 2] = palette[palette_index].r;
                }
            }
        }
    }

    return new_image;
}

static void make_output_name(const char* in, const char* operation, char* out, size_t outlen) {
    const char* ext = strrchr(in, '.');
    if (!ext) {
        snprintf(out, outlen, "%s_out_%s.bmp", in, operation);
    }
    else {
        size_t base_len = (size_t)(ext - in);
        if (base_len + strlen(operation) + 10 >= outlen) {
            strncpy_s(out, outlen, in, outlen - 1);
            return;
        }
        memcpy(out, in, base_len);
        snprintf(out + base_len, outlen - base_len, "_out_%s.bmp", operation);
    }
}

// Process and save image with a specific operation
int processAndSave(const char* input_path, const char* operation, uint8_t* original_pixels,
    int width, int original_height, int height_abs,
    BITMAPFILEHEADER* bfh, BITMAPINFOHEADER* bih, uint8_t* full_info_header, uint32_t actual_header_size,
    uint8_t* extra_data, size_t extra_data_size) {

    char output_path[4096];
    make_output_name(input_path, operation, output_path, sizeof(output_path));

    printf("\n=== Processing: %s ===\n", operation);
    printf("Output: %s\n", output_path);

    // Copy original pixels for this operation
    size_t pixel_bytes = (size_t)width * (size_t)height_abs * 3;
    uint8_t* pixels = (uint8_t*)malloc(pixel_bytes);
    if (!pixels) {
        fprintf(stderr, "Out of memory for operation: %s\n", operation);
        return 0;
    }
    memcpy(pixels, original_pixels, pixel_bytes);

    // Variables for scaling operations
    int final_width = width;
    int final_height = height_abs;
    uint8_t* final_pixels = pixels;
    int is_scaled = 0;

    // Apply the operation
    if (strcmp(operation, "grayscale") == 0) {
        applyColorTransform(pixels, width, height_abs, grayscaleMatrix);
    }
    else if (strcmp(operation, "invert") == 0) {
        applyColorInversion(pixels, width, height_abs);
    }
    else if (strcmp(operation, "sepia") == 0) {
        applyColorTransform(pixels, width, height_abs, sepiaMatrix);
    }
    else if (strcmp(operation, "blur") == 0) {
        float blurKernel[3][3] = {
            {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f},
            {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f},
            {1.0f / 3.0f, 1.0f / 3.0f, 1.0f / 3.0f}
        };
        applyConvolution(pixels, width, height_abs, blurKernel, 1.0f);
    }
    else if (strcmp(operation, "sharpen") == 0) {
        float sharpKernel[3][3] = {
            { 0.0f, -1.0f,  0.0f},
            {-1.0f,  5.0f, -1.0f},
            { 0.0f, -1.0f,  0.0f}
        };
        applyConvolution(pixels, width, height_abs, sharpKernel, 1.0f);
    }
    else if (strcmp(operation, "ridge") == 0) {
        float ridgeKernel[3][3] = {
            { 0.0f, -1.0f,  0.0f},
            {-1.0f,  4.0f, -1.0f},
            { 0.0f, -1.0f,  0.0f}
        };
        applyConvolution(pixels, width, height_abs, ridgeKernel, 1.0f);
    }
    else if (strcmp(operation, "downscale") == 0) {
        final_pixels = downscaleImage(pixels, width, height_abs, &final_width, &final_height);
        if (!final_pixels) {
            fprintf(stderr, "Failed to downscale image\n");
            free(pixels);
            return 0;
        }
        is_scaled = 1;
        printf("Downscaled from %dx%d to %dx%d\n", width, height_abs, final_width, final_height);
    }
    else if (strcmp(operation, "upscale") == 0) {
        final_pixels = upscaleImage(pixels, width, height_abs, &final_width, &final_height);
        if (!final_pixels) {
            fprintf(stderr, "Failed to upscale image\n");
            free(pixels);
            return 0;
        }
        is_scaled = 1;
        printf("Upscaled from %dx%d to %dx%d\n", width, height_abs, final_width, final_height);
    }
    else if (strcmp(operation, "dct") == 0) {
        final_pixels = applyDCTCompression(pixels, width, height_abs, &final_width, &final_height, 0.7f);
        if (!final_pixels) {
            fprintf(stderr, "Failed to apply DCT compression\n");
            free(pixels);
            return 0;
        }
        is_scaled = 1;
        printf("Applied DCT compression (quality: 0.7), new size: %dx%d\n", final_width, final_height);
    }
    else if (strcmp(operation, "bc1") == 0) {
        final_pixels = applyBC1Compression(pixels, width, height_abs, &final_width, &final_height);
        if (!final_pixels) {
            fprintf(stderr, "Failed to apply BC1 compression\n");
            free(pixels);
            return 0;
        }
        is_scaled = 1;
        printf("Applied BC1/DXT1 compression, new size: %dx%d\n", final_width, final_height);
    }

    // Write output file
    FILE* out = fopen(output_path, "wb");
    if (!out) {
        char errbuf[256];
        strerror_s(errbuf, sizeof(errbuf), errno);
        fprintf(stderr, "Failed to open output: %s\n", errbuf);
        free(pixels);
        if (is_scaled && final_pixels != pixels) free(final_pixels);
        return 0;
    }

    // Update dimensions in header if scaled
    int final_row_bytes_unpadded = final_width * 3;
    int final_row_bytes_padded = (final_row_bytes_unpadded + 3) & ~3;
    uint32_t image_size = (uint32_t)(final_row_bytes_padded * final_height);

    // Create a copy of headers for this file
    BITMAPFILEHEADER bfh_copy = *bfh;
    BITMAPINFOHEADER bih_copy = *bih;

    bih_copy.biWidth = final_width;
    bih_copy.biHeight = (original_height > 0) ? final_height : -final_height;
    bih_copy.biSizeImage = image_size;
    bfh_copy.bfSize = bfh_copy.bfOffBits + image_size;

    // Write headers
    fwrite(&bfh_copy, sizeof(bfh_copy), 1, out);

    // Update the full header with new dimensions
    uint8_t* header_copy = (uint8_t*)malloc(actual_header_size);
    memcpy(header_copy, full_info_header, actual_header_size);
    memcpy(header_copy, &bih_copy, sizeof(bih_copy));
    fwrite(header_copy, 1, actual_header_size, out);
    free(header_copy);

    // Write extra data
    if (extra_data && extra_data_size > 0) {
        fwrite(extra_data, 1, extra_data_size, out);
    }

    // Write pixel data
    uint8_t* out_row = (uint8_t*)malloc(final_row_bytes_padded);
    if (!out_row) {
        fprintf(stderr, "Out of memory (out_row)\n");
        fclose(out);
        free(pixels);
        if (is_scaled && final_pixels != pixels) free(final_pixels);
        return 0;
    }
    memset(out_row + final_row_bytes_unpadded, 0, final_row_bytes_padded - final_row_bytes_unpadded);

    if (original_height > 0) {
        for (int row = final_height - 1; row >= 0; row--) {
            memcpy(out_row, final_pixels + (size_t)row * final_row_bytes_unpadded, final_row_bytes_unpadded);
            fwrite(out_row, 1, final_row_bytes_padded, out);
        }
    }
    else {
        for (int row = 0; row < final_height; row++) {
            memcpy(out_row, final_pixels + (size_t)row * final_row_bytes_unpadded, final_row_bytes_unpadded);
            fwrite(out_row, 1, final_row_bytes_padded, out);
        }
    }

    free(out_row);
    fclose(out);
    free(pixels);
    if (is_scaled && final_pixels != pixels) free(final_pixels);

    printf("Success!\n");
    return 1;
}

int main() {
    // Hardcoded input path - CHANGE THIS TO YOUR ACTUAL PATH
    const char* input_path = "C:\\Users\\asus\\Downloads\\ph.bmp";

    printf("========================================\n");
    printf("BMP Image Processor - All Operations\n");
    printf("========================================\n");
    printf("Input: %s\n", input_path);

    FILE* f = fopen(input_path, "rb");
    if (!f) {
        perror("fopen input");
        printf("Error: Could not open input file. Please check the path: %s\n", input_path);
        return 2;
    }

    BITMAPFILEHEADER bfh;
    if (fread(&bfh, sizeof(bfh), 1, f) != 1) {
        fprintf(stderr, "Failed to read BMP file header\n");
        fclose(f);
        return 3;
    }
    if (bfh.bfType != 0x4D42) {
        fprintf(stderr, "Not a BMP file (bfType=0x%04X)\n", bfh.bfType);
        fclose(f);
        return 4;
    }

    // Read info header size first
    uint32_t header_size;
    if (fread(&header_size, sizeof(header_size), 1, f) != 1) {
        fprintf(stderr, "Failed to read info header size\n");
        fclose(f);
        return 5;
    }

    // Seek back to read the full header
    fseek(f, 14, SEEK_SET);

    // Allocate memory for the full header
    uint8_t* full_info_header = (uint8_t*)malloc(header_size);
    if (!full_info_header) {
        fprintf(stderr, "Out of memory for info header\n");
        fclose(f);
        return 6;
    }

    // Read the full header
    if (fread(full_info_header, 1, header_size, f) != header_size) {
        fprintf(stderr, "Failed to read full info header\n");
        free(full_info_header);
        fclose(f);
        return 7;
    }

    BITMAPINFOHEADER bih;
    memcpy(&bih, full_info_header, sizeof(bih));

    if (bih.biBitCount != 24) {
        fprintf(stderr, "Only 24-bit BMP supported (biBitCount=%u)\n", (unsigned)bih.biBitCount);
        free(full_info_header);
        fclose(f);
        return 8;
    }

    int width = bih.biWidth;
    int height = bih.biHeight;
    int height_abs = (height < 0) ? -height : height;
    int original_height = height;

    printf("Image dimensions: %dx%d pixels\n", width, height_abs);

    // Read extra data between headers and pixels
    size_t header_end = 14 + bih.biSize;
    size_t extra_data_size = 0;
    uint8_t* extra_data = NULL;

    if (bfh.bfOffBits > header_end) {
        extra_data_size = bfh.bfOffBits - header_end;
    }

    if (extra_data_size > 0 && extra_data_size < 10000000) {
        extra_data = (uint8_t*)malloc(extra_data_size);
        if (!extra_data) {
            fprintf(stderr, "Out of memory for extra data\n");
            free(full_info_header);
            fclose(f);
            return 9;
        }

        size_t bytes_read = fread(extra_data, 1, extra_data_size, f);

        if (bytes_read != extra_data_size) {
            fprintf(stderr, "Warning: Failed to read all extra data (%zu/%zu bytes)\n",
                bytes_read, extra_data_size);
        }
    }

    int row_bytes_unpadded = width * 3;
    int row_bytes_padded = (row_bytes_unpadded + 3) & ~3;

    size_t pixel_bytes = (size_t)width * (size_t)height_abs * 3;
    uint8_t* pixels = (uint8_t*)malloc(pixel_bytes);
    if (!pixels) {
        fprintf(stderr, "Out of memory\n");
        if (extra_data) free(extra_data);
        free(full_info_header);
        fclose(f);
        return 10;
    }

    if (fseek(f, (long)bfh.bfOffBits, SEEK_SET) != 0) {
        fprintf(stderr, "Failed to seek to pixel data (bfOffBits=%u)\n", (unsigned)bfh.bfOffBits);
        free(pixels);
        if (extra_data) free(extra_data);
        free(full_info_header);
        fclose(f);
        return 11;
    }

    uint8_t* rowbuf = (uint8_t*)malloc(row_bytes_padded);
    if (!rowbuf) {
        fprintf(stderr, "Out of memory (row)\n");
        free(pixels);
        if (extra_data) free(extra_data);
        free(full_info_header);
        fclose(f);
        return 12;
    }

    for (int row = 0; row < height_abs; row++) {
        if (fread(rowbuf, 1, row_bytes_padded, f) != (size_t)row_bytes_padded) {
            fprintf(stderr, "Failed to read BMP pixel row %d\n", row);
            free(rowbuf);
            free(pixels);
            if (extra_data) free(extra_data);
            free(full_info_header);
            fclose(f);
            return 13;
        }
        int dest_row = (height > 0) ? (height_abs - 1 - row) : row;
        memcpy(pixels + (size_t)dest_row * row_bytes_unpadded, rowbuf, row_bytes_unpadded);
    }

    free(rowbuf);
    fclose(f);

    printf("Successfully loaded image!\n");

    // Process all operations
    const char* operations[] = {
        "grayscale",
        "invert", 
        "sepia",
        "blur",
        "sharpen",
        "ridge",
        "downscale",
        "upscale",
        "dct",
		"bc1"
    };
    
    int num_operations = sizeof(operations) / sizeof(operations[0]);
    int success_count = 0;

    for (int i = 0; i < num_operations; i++) {
        success_count += processAndSave(input_path, operations[i], pixels, 
                                      width, original_height, height_abs,
                                      &bfh, &bih, full_info_header, header_size,
                                      extra_data, extra_data_size);
    }

    printf("\n========================================\n");
    printf("Processing complete! %d/%d operations successful.\n", success_count, num_operations);
    printf("Press Enter to exit...");
    getchar();

    free(pixels);
    if (extra_data) free(extra_data);
    free(full_info_header);

    return 0;
}

