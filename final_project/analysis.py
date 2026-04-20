
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
from pathlib import Path
import re

def find_samples_in_directory(directory="."):
    """
    Scrapes the target directory for images and extracts the soap concentration
    from the filename. Expects files to end with e.g. '_5p.png', '10p.jpg'.
    """
    samples = []
    valid_extensions = {".png", ".jpg", ".jpeg"}
    
    for file_path in Path(directory).iterdir():
        if file_path.suffix.lower() in valid_extensions:
            # Regex to find digits right before 'p' and the file extension
            # e.g., 'cam1_lapse2_5p.jpg' -> extracts '5'
            match = re.search(r'(\d+)p$', file_path.stem.lower())
            
            if match:
                conc = int(match.group(1))
                samples.append({
                    "image_path": str(file_path),
                    "concentration": conc,
                    "label": f"{conc}% soap",
                })
            else:
                print(f"Skipping {file_path.name}: Could not find concentration pattern (e.g., '_5p')")
                
    # Sort samples by concentration so plots and tables are ordered
    samples.sort(key=lambda x: x["concentration"])
    return samples

def detect_plate(gray: np.ndarray):

    h, w = gray.shape
    min_dim = min(h, w)
    img_cx, img_cy = w / 2, h / 2

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=min_dim // 2,
        param1=100,
        param2=25,
        minRadius=int(min_dim * 0.49),
        maxRadius=int(min_dim * 0.6),
    )

    if circles is None:
        raise RuntimeError("Could not detect the petri dish plate.")

    best = min(
        circles[0],
        key=lambda c: (c[0] - img_cx) ** 2 + (c[1] - img_cy) ** 2,
    )
    return float(best[0]), float(best[1]), float(best[2])


def detect_wafer(gray_image: np.ndarray, plate_cx: float, plate_cy: float, plate_r: float):
    """
    Detect the soap wafer as the brightest compact blob inside the plate.
    """
    search_radius = int(plate_r * 0.25)
    
    # Create a blank mask and draw a white circle at the plate's geometric center
    mask = np.zeros_like(gray_image)
    cv2.circle(mask, (int(plate_cx), int(plate_cy)), search_radius, 255, -1)
    
    # Apply the mask so everything outside the center zone becomes pitch black
    masked_gray = cv2.bitwise_and(gray_image, gray_image, mask=mask)
    
    # 2. Find the brightest spots (the white wafer)
    # Extract only the pixels inside our search zone to calculate a dynamic threshold
    center_pixels = gray_image[mask == 255]
    if len(center_pixels) == 0:
        return plate_cx, plate_cy # Fallback if something goes wrong
        
    # Isolate the top 2% brightest pixels in this center zone
    thresh_val = np.percentile(center_pixels, 98)
    _, binary = cv2.threshold(masked_gray, thresh_val, 255, cv2.THRESH_BINARY)
    
    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.dilate(binary, kernel)
    
    # cv2.imshow("bin", binary)
    
    # 3. Find contours of these bright white blobs
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # cv2.imshow("contours", cv2.drawContours(gray_image, contours, -1, (0, 255, 0), 3))
    
    if not contours:
        return plate_cx, plate_cy # Fallback to plate center
        
    # 4. Assume the largest bright blob is our wafer
    largest_contour = max(contours, key=cv2.contourArea)
    
    (wafer_cx, wafer_cy), radius = cv2.minEnclosingCircle(largest_contour)
    
    
    # Calculate the exact center (centroid) of the detected wafer
    # M = cv2.moments(largest_contour)
    # if M["m00"] != 0:
    #     wafer_cx = int(M["m10"] / M["m00"])
    #     wafer_cy = int(M["m01"] / M["m00"])
    
    # cv2.imshow("cir", cv2.circle(gray_image, (int(wafer_cx), int(wafer_cy)), int(radius), (255,0,255)))
    return wafer_cx, wafer_cy, radius

def detect_zoi(center, radius, img):
    
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    with_circle = cv2.circle(grey, center, radius + 5, np.average(grey), -1)
    
    _ ,th = cv2.threshold(with_circle, np.average(grey) + 30, 255, cv2.THRESH_BINARY)
    kernel = np.ones((2,2), np.uint8)

    # Apply erosion
    eroded_img = cv2.erode(th, kernel, iterations=1)
    
    # blurred = cv2.GaussianBlur(cv2.cvtColor(th, cv2.COLOR_GRAY2BGR), (5, 5), 0)
    
    rad = 1
    while(True):
        h, w = th.shape[:2]

        # 2. Create a blank black mask (same size as image)
        mask = np.zeros((h, w), dtype=np.uint8)

        # 3. Draw a white (255) filled (-1 thickness) circle on the mask
        # Syntax: cv2.circle(img, center, radius, color, thickness)
        cv2.circle(mask, center, rad, 255, -1)

        # 4. Apply the mask using bitwise_and
        masked_image = cv2.bitwise_and(eroded_img, eroded_img, mask=mask)
        
        # cv2.imshow("mask", masked_image)
        # cv2.waitKey()
        count = np.count_nonzero(masked_image)
        area = np.pi * rad * rad
        if (count / area) > .05:
            break
        rad += 1
    
    # cv2.imshow("th", th)
    # cv2.imshow("circ", with_circle)
    # cv2.waitKey()
    return rad
    

def analyse_image(image_path: str):
    """Full analysis pipeline for one image."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    px, py, pr = detect_plate(gray)
    
    
    wx, wy, wr = detect_wafer(gray, px, py, pr)
    
    circle = cv2.circle(img, (int(wx), int(wy)), int(wr), (255,0,0))
    
    # cv2.imshow("asf", circle)
    # cv2.waitKey()

    zoi_r = detect_zoi((int(wx), int(wy)), int(wr), img)
    
    # print(zoi_r)

    proportion_dead = (zoi_r / pr) ** 2

    return {
        "image":          img,
        "plate":          (px, py, pr),
        "wafer":          (wx, wy, wr),
        "zoi_radius":     zoi_r,
        "proportion_dead": proportion_dead,
    }


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------
def annotate_image(result: dict) -> np.ndarray:
    vis = cv2.cvtColor(result["image"], cv2.COLOR_BGR2RGB).copy()
    px, py, pr = result["plate"]
    wx, wy, wr = result["wafer"]
    zr = result["zoi_radius"]

    # Draw the regions
    cv2.circle(vis, (int(px), int(py)), int(pr), (0, 200, 0),   2)  # Plate - Green
    cv2.circle(vis, (int(wx), int(wy)), int(wr), (255, 120, 0), 2)  # Wafer - Orange
    cv2.circle(vis, (int(wx), int(wy)), int(zr), (220, 30, 30), 2)  # ZOI - Red

    return vis


def plot_results(samples, results, output_path="zone_of_inhibition_results.png"):
    n = len(samples)
    if n == 0:
        print("No results to plot.")
        return
        
    fig = plt.figure(figsize=(5 * n, 10))
    # 2 rows now: Images on top, Scatter plot on the bottom
    gs  = fig.add_gridspec(2, n, height_ratios=[2, 2], hspace=0.3, wspace=0.3)

    concentrations = [s["concentration"] for s in samples]
    proportions    = [r["proportion_dead"] for r in results]

    for i, (sample, result) in enumerate(zip(samples, results)):
        conc  = sample["concentration"]
        label = sample.get("label", f"{conc}")

        # ── annotated image ──────────────────────────────────────────────
        ax_img = fig.add_subplot(gs[0, i])
        ax_img.imshow(annotate_image(result))
        ax_img.set_title(
            f"{label}\nDead area: {result['proportion_dead']*100:.1f}%",
            fontsize=10,
        )
        ax_img.axis("off")

    # ── concentration vs ZOI scatter ───────────────────────────────────────
    ax_plot = fig.add_subplot(gs[1, :])
    ax_plot.scatter(concentrations, [p * 100 for p in proportions],
                    s=140, zorder=5, c="firebrick", edgecolors="navy")

    for conc, prop, sample in zip(concentrations, proportions, samples):
        label = sample.get("label", f"{conc}")
        ax_plot.annotate(
            label,
            (conc, prop * 100),
            textcoords="offset points", xytext=(6, 4), fontsize=9,
        )

    # Add trendline if there are enough points
    if len(concentrations) >= 2:
        z = np.polyfit(concentrations, [p * 100 for p in proportions], 1)
        x_fit = np.linspace(min(concentrations), max(concentrations), 200)
        ax_plot.plot(x_fit, np.polyval(z, x_fit), "r--", lw=1.5, label="Trend")
        ax_plot.legend()

    ax_plot.set_xlabel("Soap concentration", fontsize=12)
    ax_plot.set_ylabel("Zone of Inhibition (% of plate area)", fontsize=12)
    ax_plot.set_title("Soap Concentration vs Zone of Inhibition", fontsize=13, fontweight="bold")
    ax_plot.grid(True, alpha=0.3)
    ax_plot.set_ylim(bottom=0)

    fig.suptitle(
        "Zone of Inhibition Analysis\n"
        "● Green = plate  ● Orange = wafer  ● Red = ZOI boundary",
        fontsize=11, y=0.98,
    )

    plt.savefig(output_path, dpi=130, bbox_inches="tight")
    print(f"Figure saved → {output_path}")
    return fig


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

OUTPUT_FIGURE = "zone_of_inhibition_results.png"

if __name__ == "__main__":
    print("Scraping directory for images...")
    SAMPLES = find_samples_in_directory(".")
    
    if not SAMPLES:
        print("No valid images found matching the '*_XXp.[png|jpg]' pattern in the current directory.")
    else:
        results = []
        print(f"\nFound {len(SAMPLES)} images to process.\n")
        print(f"{'Image':<30} {'Plate r':>8} {'Wafer r':>8} {'ZOI r':>8} {'% dead':>8} {'ZOI?':>6}")
        print("-" * 78)

        for sample in SAMPLES:
            path   = sample["image_path"]
            # if ("100p" not in path):
            #     continue
            result = analyse_image(path)
            results.append(result)

            px, py, pr = result["plate"]
            wx, wy, wr = result["wafer"]
            print(
                f"{Path(path).name:<30} "
                f"{pr:>8.1f} "
                f"{wr:>8.1f} "
                f"{result['zoi_radius']:>8.1f} "
                f"{result['proportion_dead']*100:>7.1f}% "
                # f"{'yes' if result['halo_detected'] else 'no':>6}"
            )

        plot_results(SAMPLES, results, OUTPUT_FIGURE)