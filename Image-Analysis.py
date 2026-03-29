# ============================================
# MINI NUMPY IMAGE PROJECT - COLAB READY
# ============================================
# It creates a small synthetic grayscale image,
# analyzes it, transforms it, and visualizes results.
# ============================================

import numpy as np
import matplotlib.pyplot as plt


class ImageAnalyzer:
    def __init__(self, image: np.ndarray):
        self.image = image

    def validate(self) -> None:
        if not isinstance(self.image, np.ndarray):
            raise TypeError("image must be a NumPy array")
        if self.image.ndim != 2:
            raise ValueError("This project expects a 2D grayscale image")
        if self.image.size == 0:
            raise ValueError("image cannot be empty")

    def stats(self) -> dict:
        return {
            "shape": self.image.shape,
            "dtype": self.image.dtype,
            "min": int(np.min(self.image)),
            "max": int(np.max(self.image)),
            "mean": float(np.mean(self.image)),
            "std": float(np.std(self.image)),
            "sum": int(np.sum(self.image)),
        }

    def count_pixels_above(self, threshold: int) -> int:
        return int(np.sum(self.image > threshold))

    def count_pixels_below(self, threshold: int) -> int:
        return int(np.sum(self.image < threshold))

    def dark_mask(self, threshold: int = 80) -> np.ndarray:
        return self.image < threshold

    def bright_mask(self, threshold: int = 180) -> np.ndarray:
        return self.image > threshold

    def row_means(self) -> np.ndarray:
        return np.mean(self.image, axis=1)

    def col_means(self) -> np.ndarray:
        return np.mean(self.image, axis=0)


class ImageProcessor:
    def __init__(self, image: np.ndarray):
        self.image = image.astype(np.int16)  # safe temporary type for math

    def brighten(self, value: int = 30) -> np.ndarray:
        return np.clip(self.image + value, 0, 255).astype(np.uint8)

    def darken(self, value: int = 30) -> np.ndarray:
        return np.clip(self.image - value, 0, 255).astype(np.uint8)

    def invert(self) -> np.ndarray:
        return (255 - self.image).astype(np.uint8)

    def binary_threshold(self, threshold: int = 100) -> np.ndarray:
        return np.where(self.image > threshold, 255, 0).astype(np.uint8)

    def normalize_0_1(self) -> np.ndarray:
        return self.image.astype(np.float32) / 255.0

    def crop_center(self, crop_h: int = 2, crop_w: int = 2) -> np.ndarray:
        h, w = self.image.shape
        start_h = max((h - crop_h) // 2, 0)
        start_w = max((w - crop_w) // 2, 0)
        end_h = start_h + crop_h
        end_w = start_w + crop_w
        return self.image[start_h:end_h, start_w:end_w].astype(np.uint8)

    def horizontal_difference(self) -> np.ndarray:
        return np.abs(self.image[:, 1:] - self.image[:, :-1]).astype(np.uint8)

    def vertical_difference(self) -> np.ndarray:
        return np.abs(self.image[1:, :] - self.image[:-1, :]).astype(np.uint8)

    def blur_mean_3x3(self) -> np.ndarray:
        padded = np.pad(self.image, pad_width=1, mode="edge")
        output = np.zeros_like(self.image, dtype=np.float32)

        for i in range(self.image.shape[0]):
            for j in range(self.image.shape[1]):
                window = padded[i:i+3, j:j+3]
                output[i, j] = np.mean(window)

        return np.clip(output, 0, 255).astype(np.uint8)


def print_section(title: str) -> None:
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def show_image(title: str, image: np.ndarray) -> None:
    plt.figure(figsize=(4, 4))
    plt.imshow(image, cmap="gray", vmin=0, vmax=255)
    plt.title(title)
    plt.axis("off")
    plt.show()


def show_histogram(title: str, image: np.ndarray) -> None:
    plt.figure(figsize=(6, 4))
    plt.hist(image.ravel(), bins=16)
    plt.title(title)
    plt.xlabel("Pixel Value")
    plt.ylabel("Count")
    plt.show()


def main():
    # ------------------------------------------------
    # 1) Create a synthetic grayscale image
    # ------------------------------------------------
    image = np.array([
        [10,  20,  30, 200, 210, 220],
        [40,  50,  60, 190, 200, 210],
        [70,  80,  90, 180, 190, 200],
        [100, 110, 120, 170, 180, 190],
        [130, 140, 150, 160, 170, 180],
        [160, 170, 180, 150, 140, 130]
    ], dtype=np.uint8)

    analyzer = ImageAnalyzer(image)
    analyzer.validate()

    processor = ImageProcessor(image)

    # ------------------------------------------------
    # 2) Analyze image
    # ------------------------------------------------
    print_section("ORIGINAL IMAGE")
    print(image)

    print_section("IMAGE STATISTICS")
    stats = analyzer.stats()
    for key, value in stats.items():
        print(f"{key}: {value}")

    print_section("MASKS AND COUNTS")
    dark_mask = analyzer.dark_mask(threshold=80)
    bright_mask = analyzer.bright_mask(threshold=180)

    print("Dark mask (< 80):")
    print(dark_mask)

    print("\nBright mask (> 180):")
    print(bright_mask)

    print("\nPixels above 100:", analyzer.count_pixels_above(100))
    print("Pixels below 80:", analyzer.count_pixels_below(80))

    print_section("ROW/COLUMN MEANS")
    print("Row means:")
    print(analyzer.row_means())

    print("\nColumn means:")
    print(analyzer.col_means())

    # ------------------------------------------------
    # 3) Process image
    # ------------------------------------------------
    brightened = processor.brighten(30)
    darkened = processor.darken(30)
    inverted = processor.invert()
    binary = processor.binary_threshold(120)
    normalized = processor.normalize_0_1()
    cropped = processor.crop_center(3, 3)
    hdiff = processor.horizontal_difference()
    vdiff = processor.vertical_difference()
    blurred = processor.blur_mean_3x3()

    # ------------------------------------------------
    # 4) Print transformed arrays
    # ------------------------------------------------
    print_section("BRIGHTENED IMAGE (+30)")
    print(brightened)

    print_section("DARKENED IMAGE (-30)")
    print(darkened)

    print_section("INVERTED IMAGE")
    print(inverted)

    print_section("BINARY IMAGE (threshold=120)")
    print(binary)

    print_section("NORMALIZED IMAGE (0 TO 1)")
    print(normalized)

    print_section("CENTER CROP (3x3)")
    print(cropped)

    print_section("HORIZONTAL DIFFERENCE")
    print(hdiff)

    print_section("VERTICAL DIFFERENCE")
    print(vdiff)

    print_section("BLURRED IMAGE (3x3 mean filter)")
    print(blurred)

    # ------------------------------------------------
    # 5) Visualize results
    # ------------------------------------------------
    show_image("Original Image", image)
    show_image("Brightened", brightened)
    show_image("Darkened", darkened)
    show_image("Inverted", inverted)
    show_image("Binary", binary)
    show_image("Blurred", blurred)
    show_image("Center Crop", cropped)

    show_histogram("Histogram - Original Image", image)
    show_histogram("Histogram - Brightened Image", brightened)

    # ------------------------------------------------
    # 6) Small ML-style feature example
    # ------------------------------------------------
    print_section("MINI ML-STYLE FEATURE MATRIX EXAMPLE")

    X = np.array([
        [170, 65],
        [180, 80],
        [160, 55],
        [175, 75],
        [172, 68]
    ], dtype=np.float32)

    print("Original feature matrix:")
    print(X)

    means = np.mean(X, axis=0)
    stds = np.std(X, axis=0)

    X_normalized = (X - means) / stds

    print("\nFeature means [height_mean, weight_mean]:")
    print(means)

    print("\nFeature std [height_std, weight_std]:")
    print(stds)

    print("\nNormalized feature matrix:")
    print(X_normalized)


if __name__ == "__main__":
    main()
```
