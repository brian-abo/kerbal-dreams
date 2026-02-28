"""
KerbalDreams Texture Generator
Generates procedural color, height, normal, and biome maps for all bodies.
Output: PNG files in GameData/KerbalDreams/Textures/PluginData/
"""
import numpy as np
from PIL import Image
import os
import math

OUT = "GameData/KerbalDreams/Textures/PluginData"
os.makedirs(OUT, exist_ok=True)

# --- Noise functions (pure numpy, no external noise lib) ---

def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)

def lerp(a, b, t):
    return a + t * (b - a)

class PerlinNoise:
    def __init__(self, seed=0):
        rng = np.random.RandomState(seed)
        self.p = rng.permutation(256).astype(np.int32)
        self.p = np.tile(self.p, 2)
        angles = rng.uniform(0, 2 * np.pi, 256)
        self.gx = np.cos(angles).astype(np.float64)
        self.gy = np.sin(angles).astype(np.float64)
        self.gx = np.tile(self.gx, 2)
        self.gy = np.tile(self.gy, 2)

    def __call__(self, x, y):
        xi = np.floor(x).astype(np.int32) & 255
        yi = np.floor(y).astype(np.int32) & 255
        xf = x - np.floor(x)
        yf = y - np.floor(y)
        u = fade(xf)
        v = fade(yf)

        aa = self.p[self.p[xi] + yi]
        ab = self.p[self.p[xi] + yi + 1]
        ba = self.p[self.p[xi + 1] + yi]
        bb = self.p[self.p[xi + 1] + yi + 1]

        x1 = lerp(self.gx[aa] * xf + self.gy[aa] * yf,
                   self.gx[ba] * (xf - 1) + self.gy[ba] * yf, u)
        x2 = lerp(self.gx[ab] * xf + self.gy[ab] * (yf - 1),
                   self.gx[bb] * (xf - 1) + self.gy[bb] * (yf - 1), u)
        return lerp(x1, x2, v)


def fbm(noise, x, y, octaves=6, lacunarity=2.0, gain=0.5):
    """Fractal Brownian Motion"""
    val = np.zeros_like(x, dtype=np.float64)
    amp = 1.0
    freq = 1.0
    for _ in range(octaves):
        val += amp * noise(x * freq, y * freq)
        freq *= lacunarity
        amp *= gain
    return val


def ridged_fbm(noise, x, y, octaves=6, lacunarity=2.0, gain=0.5):
    """Ridged multifractal noise"""
    val = np.zeros_like(x, dtype=np.float64)
    amp = 1.0
    freq = 1.0
    for _ in range(octaves):
        n = noise(x * freq, y * freq)
        n = 1.0 - np.abs(n)
        n = n * n
        val += amp * n
        freq *= lacunarity
        amp *= gain
    return val


def make_coords(w, h, scale=1.0):
    """Create seamlessly tileable spherical coordinates mapped to 2D"""
    u = np.linspace(0, 1, w, endpoint=False)
    v = np.linspace(0, 1, h, endpoint=False)
    u, v = np.meshgrid(u, v)
    # Map to spherical coordinates for seamless wrapping
    theta = u * 2 * np.pi
    phi = v * np.pi
    # Project onto 3D sphere, then take 2 pairs of coords for 2D noise
    x = np.cos(theta) * np.sin(phi) * scale
    y = np.sin(theta) * np.sin(phi) * scale
    z = np.cos(phi) * scale
    return x, y, z, u, v, phi


def spherical_noise(seed, w, h, scale, octaves=6, gain=0.5, ridged=False):
    """Generate noise on a sphere (seamless at edges)"""
    n1 = PerlinNoise(seed)
    n2 = PerlinNoise(seed + 1000)
    x, y, z, u, v, phi = make_coords(w, h, scale)
    # Use 2 independent 2D noise samples combined with sphere coords
    if ridged:
        val = ridged_fbm(n1, x + z * 0.5, y + z * 0.5, octaves=octaves, gain=gain)
    else:
        val = fbm(n1, x + z * 0.5, y + z * 0.5, octaves=octaves, gain=gain)
    # Add second layer for more variation
    val += 0.3 * fbm(n2, y + x * 0.3, z + x * 0.3, octaves=max(2, octaves - 2), gain=gain)
    return val


def normalize(arr):
    mn, mx = arr.min(), arr.max()
    if mx - mn < 1e-10:
        return np.zeros_like(arr)
    return (arr - mn) / (mx - mn)


def color_lerp(t, c1, c2):
    """Lerp between two RGB colors based on t (0-1 array)"""
    t = np.clip(t, 0, 1)
    r = c1[0] + t * (c2[0] - c1[0])
    g = c1[1] + t * (c2[1] - c1[1])
    b = c1[2] + t * (c2[2] - c1[2])
    return r, g, b


def height_to_normal(height_arr, strength=2.0):
    """Convert height map to normal map"""
    h = height_arr.astype(np.float64) / 255.0
    # Sobel-like derivatives
    dx = np.roll(h, -1, axis=1) - np.roll(h, 1, axis=1)
    dy = np.roll(h, -1, axis=0) - np.roll(h, 1, axis=0)
    dx *= strength
    dy *= strength
    nz = np.ones_like(h)
    length = np.sqrt(dx * dx + dy * dy + nz * nz)
    nx = (-dx / length * 0.5 + 0.5) * 255
    ny = (-dy / length * 0.5 + 0.5) * 255
    nzz = (nz / length * 0.5 + 0.5) * 255
    normal = np.stack([nx, ny, nzz], axis=-1).astype(np.uint8)
    return normal


def save_png(arr, path):
    """Save numpy array as PNG"""
    if arr.ndim == 2:
        img = Image.fromarray(arr.astype(np.uint8), mode='L')
    else:
        img = Image.fromarray(arr.astype(np.uint8), mode='RGB')
    img.save(path)
    print(f"  Saved: {path}")


# ============================================================================
# URVA (Star) - just a bright yellow-white disc
# ============================================================================
def gen_urva():
    print("Generating Urva (star)...")
    w, h = 2048, 1024
    n = spherical_noise(1000, w, h, 4.0, octaves=8, gain=0.5)
    n = normalize(n)
    # Hot star surface: yellow-white with slightly darker convection cells
    r = (0.95 + 0.05 * n) * 255
    g = (0.85 + 0.1 * n) * 255
    b = (0.55 + 0.3 * n) * 255
    color = np.stack([r, g, b], axis=-1).astype(np.uint8)
    save_png(color, f"{OUT}/Urva_color.png")


# ============================================================================
# USIN - Dense super-Kerbin: golden mesas on black rock
# ============================================================================
def gen_usin():
    print("Generating Usin (golden mesas on black rock)...")
    w, h = 4096, 2048

    # Height: mesa-like terrain using ridged noise + thresholding
    height_raw = spherical_noise(2000, w, h, 5.0, octaves=7, gain=0.5, ridged=True)
    height_raw = normalize(height_raw)

    # Create mesa effect: plateau at certain heights
    mesa = np.copy(height_raw)
    # Flatten tops of mesas
    mesa_threshold = 0.55
    mesa[mesa > mesa_threshold] = mesa_threshold + (mesa[mesa > mesa_threshold] - mesa_threshold) * 0.15
    mesa = normalize(mesa)

    height = (mesa * 255).astype(np.uint8)

    # Color: black rock base, golden where elevated (mesas)
    detail = spherical_noise(2001, w, h, 12.0, octaves=5, gain=0.4)
    detail = normalize(detail)

    t = np.clip((mesa - 0.35) / 0.3, 0, 1)  # transition zone
    # Black rock: very dark grey
    r_base, g_base, b_base = 25, 22, 18
    # Gold: rich metallic gold
    r_gold, g_gold, b_gold = 205, 170, 45

    r = r_base + t * (r_gold - r_base) + detail * 20
    g = g_base + t * (g_gold - g_base) + detail * 15
    b = b_base + t * (b_gold - b_base) + detail * 8

    # Add some variation - darker cracks/veins
    crack = spherical_noise(2002, w, h, 20.0, octaves=4, gain=0.6)
    crack = normalize(crack)
    crack_mask = (crack < 0.2).astype(np.float64) * 0.4
    r *= (1 - crack_mask)
    g *= (1 - crack_mask)
    b *= (1 - crack_mask)

    color = np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(np.uint8)

    # Normal map
    normal = height_to_normal(height, strength=3.0)

    # Biome map: Golden Mesas (0.85,0.7,0.15), Obsidian Flats (0.1,0.1,0.1),
    #            Crater Fields (0.3,0.25,0.1), Dark Ridges (0.2,0.15,0.05)
    biome = np.zeros((h, w, 3), dtype=np.uint8)
    # Craters from noise
    crater_noise = spherical_noise(2003, w, h, 8.0, octaves=4, gain=0.5)
    crater_noise = normalize(crater_noise)

    # Default: Obsidian Flats (low areas)
    biome[:, :] = [26, 26, 26]  # 0.1, 0.1, 0.1
    # Dark Ridges (medium height, steep)
    mask_ridges = (mesa > 0.3) & (mesa < 0.5)
    biome[mask_ridges] = [51, 38, 13]  # 0.2, 0.15, 0.05
    # Crater Fields
    mask_craters = crater_noise < 0.25
    biome[mask_craters] = [77, 64, 26]  # 0.3, 0.25, 0.1
    # Golden Mesas (high areas)
    mask_mesa = mesa > 0.5
    biome[mask_mesa] = [217, 179, 38]  # 0.85, 0.7, 0.15

    save_png(color, f"{OUT}/Usin_color.png")
    save_png(height, f"{OUT}/Usin_height.png")
    save_png(normal, f"{OUT}/Usin_normal.png")
    save_png(biome, f"{OUT}/Usin_biome.png")


# ============================================================================
# MYUN - Global ocean world with rings
# ============================================================================
def gen_myun():
    print("Generating Myun (global ocean world)...")
    w, h = 4096, 2048

    # Height: all underwater, gentle ocean floor variation
    height_raw = spherical_noise(3000, w, h, 4.0, octaves=6, gain=0.5)
    height_raw = normalize(height_raw)
    # Keep everything low (below sea level conceptually)
    height = (height_raw * 180).astype(np.uint8)  # max ~70% height

    # Color: deep ocean blues and teals
    detail = spherical_noise(3001, w, h, 6.0, octaves=5, gain=0.4)
    detail = normalize(detail)

    depth_factor = normalize(height_raw)

    # Shallow: lighter turquoise. Deep: dark navy
    r = 15 + depth_factor * 40 + detail * 15
    g = 55 + depth_factor * 80 + detail * 25
    b = 130 + depth_factor * 60 + detail * 30

    # Subtle current/wave patterns
    wave = spherical_noise(3002, w, h, 15.0, octaves=3, gain=0.3)
    wave = normalize(wave)
    r += wave * 8
    g += wave * 12
    b += wave * 15

    color = np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(np.uint8)

    normal = height_to_normal(height, strength=1.0)

    # Biome map: Open Ocean, Deep Trench, Shallows, Ring Shadow Zone
    biome = np.zeros((h, w, 3), dtype=np.uint8)
    x, y, z, u, v, phi = make_coords(w, h)

    # Default: Open Ocean
    biome[:, :] = [0, 51, 179]  # 0.0, 0.2, 0.7
    # Shallows (higher ocean floor)
    mask_shallow = height_raw > 0.6
    biome[mask_shallow] = [51, 128, 204]  # 0.2, 0.5, 0.8
    # Deep Trench (very low)
    mask_deep = height_raw < 0.2
    biome[mask_deep] = [0, 13, 77]  # 0.0, 0.05, 0.3
    # Ring Shadow Zone (equatorial band)
    lat = np.abs(phi - np.pi / 2)
    mask_ring = lat < 0.15
    biome[mask_ring] = [26, 26, 102]  # 0.1, 0.1, 0.4

    save_png(color, f"{OUT}/Myun_color.png")
    save_png(height, f"{OUT}/Myun_height.png")
    save_png(normal, f"{OUT}/Myun_normal.png")
    save_png(biome, f"{OUT}/Myun_biome.png")


# ============================================================================
# RENN - Grey asteroid moon (12km)
# ============================================================================
def gen_renn():
    print("Generating Renn (grey asteroid)...")
    w, h = 1024, 512

    height_raw = spherical_noise(4000, w, h, 3.0, octaves=6, gain=0.6, ridged=True)
    height_raw = normalize(height_raw)
    height = (height_raw * 255).astype(np.uint8)

    detail = spherical_noise(4001, w, h, 10.0, octaves=4, gain=0.4)
    detail = normalize(detail)

    grey = 90 + height_raw * 70 + detail * 30
    r = grey + detail * 5
    g = grey
    b = grey - detail * 3

    color = np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(np.uint8)
    normal = height_to_normal(height, strength=4.0)

    save_png(color, f"{OUT}/Renn_color.png")
    save_png(height, f"{OUT}/Renn_height.png")
    save_png(normal, f"{OUT}/Renn_normal.png")


# ============================================================================
# TAVI - Rust-tinted asteroid (8km)
# ============================================================================
def gen_tavi():
    print("Generating Tavi (rust asteroid)...")
    w, h = 1024, 512

    height_raw = spherical_noise(5000, w, h, 3.5, octaves=6, gain=0.55, ridged=True)
    height_raw = normalize(height_raw)
    height = (height_raw * 255).astype(np.uint8)

    detail = spherical_noise(5001, w, h, 12.0, octaves=4, gain=0.4)
    detail = normalize(detail)

    # Rust orange-brown
    r = 120 + height_raw * 60 + detail * 35
    g = 65 + height_raw * 35 + detail * 20
    b = 35 + height_raw * 20 + detail * 10

    color = np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(np.uint8)
    normal = height_to_normal(height, strength=4.0)

    save_png(color, f"{OUT}/Tavi_color.png")
    save_png(height, f"{OUT}/Tavi_height.png")
    save_png(normal, f"{OUT}/Tavi_normal.png")


# ============================================================================
# KOSS - Dark carbonaceous asteroid (6km)
# ============================================================================
def gen_koss():
    print("Generating Koss (dark carbonaceous asteroid)...")
    w, h = 1024, 512

    height_raw = spherical_noise(6000, w, h, 3.0, octaves=5, gain=0.6, ridged=True)
    height_raw = normalize(height_raw)
    height = (height_raw * 255).astype(np.uint8)

    detail = spherical_noise(6001, w, h, 10.0, octaves=4, gain=0.4)
    detail = normalize(detail)

    # Very dark with slight purple-grey tint
    r = 40 + height_raw * 30 + detail * 15
    g = 35 + height_raw * 25 + detail * 12
    b = 45 + height_raw * 30 + detail * 18

    color = np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(np.uint8)
    normal = height_to_normal(height, strength=4.0)

    save_png(color, f"{OUT}/Koss_color.png")
    save_png(height, f"{OUT}/Koss_height.png")
    save_png(normal, f"{OUT}/Koss_normal.png")


# ============================================================================
# KENKAN - Small ice giant: magenta/red atmosphere, indigo cloud bands
# ============================================================================
def gen_kenkan():
    print("Generating Kenkan (magenta/red ice giant)...")
    w, h = 4096, 2048

    x, y, z, u, v, phi = make_coords(w, h)

    # Banded structure based on latitude
    lat = (phi / np.pi)  # 0 at north pole, 1 at south pole

    # Base magenta-red color
    base_r = 180 + 40 * np.sin(lat * np.pi * 12)
    base_g = 40 + 25 * np.sin(lat * np.pi * 12 + 1.5)
    base_b = 90 + 50 * np.sin(lat * np.pi * 12 + 0.8)

    # Indigo cloud bands
    band_noise = spherical_noise(7000, w, h, 3.0, octaves=4, gain=0.4)
    band_noise = normalize(band_noise)

    # Create distinct bands
    band_pattern = np.sin(lat * np.pi * 16 + band_noise * 2) * 0.5 + 0.5
    band_pattern = normalize(band_pattern)

    # Where bands are dark -> indigo
    indigo_mask = band_pattern
    base_r = base_r * (1 - indigo_mask * 0.5) + indigo_mask * 45 * 0.5
    base_g = base_g * (1 - indigo_mask * 0.4) + indigo_mask * 25 * 0.4
    base_b = base_b * (1 - indigo_mask * 0.3) + indigo_mask * 120 * 0.5

    # Storm detail
    storm = spherical_noise(7001, w, h, 8.0, octaves=6, gain=0.45)
    storm = normalize(storm)
    base_r += storm * 25
    base_g += storm * 10
    base_b += storm * 20

    # Swirl effect: offset x based on latitude for zonal wind appearance
    swirl = spherical_noise(7002, w, h, 6.0, octaves=3, gain=0.3)
    swirl = normalize(swirl)
    base_r += swirl * 15 * np.sin(lat * np.pi * 8)
    base_g += swirl * 8 * np.sin(lat * np.pi * 8)
    base_b += swirl * 12 * np.sin(lat * np.pi * 8)

    color = np.stack([np.clip(base_r, 0, 255), np.clip(base_g, 0, 255), np.clip(base_b, 0, 255)], axis=-1).astype(np.uint8)

    save_png(color, f"{OUT}/Kenkan_color.png")


# ============================================================================
# MOZAZ - Ammonia ocean moon: brown slush, north ice cap
# ============================================================================
def gen_mozaz():
    print("Generating Mozaz (ammonia ocean moon)...")
    w, h = 2048, 1024

    height_raw = spherical_noise(8000, w, h, 5.0, octaves=6, gain=0.5)
    height_raw = normalize(height_raw)

    # Add ridges
    ridges = spherical_noise(8001, w, h, 7.0, octaves=5, gain=0.5, ridged=True)
    ridges = normalize(ridges)
    combined = height_raw * 0.6 + ridges * 0.4
    combined = normalize(combined)

    # Lower overall to allow ocean coverage
    height = (combined * 200 + 20).astype(np.uint8)

    # Color
    detail = spherical_noise(8002, w, h, 10.0, octaves=5, gain=0.4)
    detail = normalize(detail)

    x, y, z, u, v, phi = make_coords(w, h)
    lat = phi / np.pi  # 0=north, 1=south

    # Base: brown slush
    r = 140 + combined * 40 + detail * 25
    g = 105 + combined * 30 + detail * 18
    b = 55 + combined * 20 + detail * 12

    # Ammonia ocean (low areas): yellowish-brown
    ocean_mask = np.clip(1 - (combined - 0.3) / 0.2, 0, 1)
    r = r * (1 - ocean_mask * 0.3) + ocean_mask * 130 * 0.3
    g = g * (1 - ocean_mask * 0.3) + ocean_mask * 100 * 0.3
    b = b * (1 - ocean_mask * 0.4) + ocean_mask * 35 * 0.4

    # North ice cap: brilliant white, only at north pole
    ice_edge = 0.18  # ~32 degrees from north pole
    ice_blend = np.clip((ice_edge - lat) / 0.06, 0, 1)
    # Add irregular edge
    edge_noise = spherical_noise(8003, w, h, 12.0, octaves=4, gain=0.5)
    edge_noise = normalize(edge_noise)
    ice_blend = np.clip(ice_blend + (edge_noise - 0.5) * 0.3, 0, 1)

    r = r * (1 - ice_blend) + 225 * ice_blend
    g = g * (1 - ice_blend) + 235 * ice_blend
    b = b * (1 - ice_blend) + 245 * ice_blend

    color = np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(np.uint8)
    normal = height_to_normal(height, strength=2.5)

    # Biome map
    biome = np.zeros((h, w, 3), dtype=np.uint8)
    # Default: Slush Fields
    biome[:, :] = [115, 89, 51]  # 0.45, 0.35, 0.2
    # Ammonia Sea (low)
    mask_ocean = combined < 0.35
    biome[mask_ocean] = [140, 115, 51]  # 0.55, 0.45, 0.2
    # Ridgeline (high)
    mask_ridge = combined > 0.7
    biome[mask_ridge] = [140, 102, 64]  # 0.55, 0.4, 0.25
    # Polar Ice Cap
    mask_ice = ice_blend > 0.5
    biome[mask_ice] = [217, 230, 242]  # 0.85, 0.9, 0.95

    save_png(color, f"{OUT}/Mozaz_color.png")
    save_png(height, f"{OUT}/Mozaz_height.png")
    save_png(normal, f"{OUT}/Mozaz_normal.png")
    save_png(biome, f"{OUT}/Mozaz_biome.png")


# ============================================================================
# ICARI - Cold rocky world: dark red/orange tholins with green patches
# ============================================================================
def gen_icari():
    print("Generating Icari (tholin world with green patches)...")
    w, h = 2048, 1024

    height_raw = spherical_noise(9000, w, h, 5.0, octaves=7, gain=0.5)
    height_raw = normalize(height_raw)

    # Impact basins
    basins = spherical_noise(9001, w, h, 3.0, octaves=4, gain=0.6)
    basins = normalize(basins)
    basin_mask = (basins < 0.25).astype(np.float64)
    combined = height_raw * (1 - basin_mask * 0.5)
    combined = normalize(combined)

    height = (combined * 255).astype(np.uint8)

    detail = spherical_noise(9002, w, h, 12.0, octaves=5, gain=0.4)
    detail = normalize(detail)

    x, y, z, u, v, phi = make_coords(w, h)
    lat = phi / np.pi

    # Base: dark red-orange tholins
    r = 130 + combined * 35 + detail * 25
    g = 55 + combined * 20 + detail * 12
    b = 20 + combined * 12 + detail * 8

    # Green patches: scattered, blob-like
    green_noise = spherical_noise(9003, w, h, 6.0, octaves=5, gain=0.5)
    green_noise = normalize(green_noise)
    green_mask = np.clip((green_noise - 0.65) / 0.15, 0, 1)

    # Green tint
    r = r * (1 - green_mask * 0.6) + green_mask * 50 * 0.6
    g = g * (1 - green_mask * 0.7) + green_mask * 120 * 0.7
    b = b * (1 - green_mask * 0.5) + green_mask * 45 * 0.5

    # Polar wastes: slightly darker, more muted
    polar = np.clip((np.abs(lat - 0.5) - 0.35) / 0.15, 0, 1)
    r *= (1 - polar * 0.25)
    g *= (1 - polar * 0.2)
    b *= (1 - polar * 0.15)

    color = np.stack([np.clip(r, 0, 255), np.clip(g, 0, 255), np.clip(b, 0, 255)], axis=-1).astype(np.uint8)
    normal = height_to_normal(height, strength=2.5)

    # Biome map
    biome = np.zeros((h, w, 3), dtype=np.uint8)
    # Default: Tholin Plains
    biome[:, :] = [140, 51, 20]  # 0.55, 0.2, 0.08
    # Green Patches
    mask_green = green_noise > 0.65
    biome[mask_green] = [38, 102, 38]  # 0.15, 0.4, 0.15
    # Impact Basins
    mask_basin = basins < 0.25
    biome[mask_basin] = [89, 31, 13]  # 0.35, 0.12, 0.05
    # Polar Wastes
    mask_polar = np.abs(lat - 0.5) > 0.38
    biome[mask_polar] = [102, 64, 38]  # 0.4, 0.25, 0.15

    save_png(color, f"{OUT}/Icari_color.png")
    save_png(height, f"{OUT}/Icari_height.png")
    save_png(normal, f"{OUT}/Icari_normal.png")
    save_png(biome, f"{OUT}/Icari_biome.png")


# ============================================================================
# Generate all
# ============================================================================
if __name__ == "__main__":
    print("=== KerbalDreams Texture Generator ===\n")
    gen_urva()
    gen_usin()
    gen_myun()
    gen_renn()
    gen_tavi()
    gen_koss()
    gen_kenkan()
    gen_mozaz()
    gen_icari()
    print("\n=== All textures generated! ===")
