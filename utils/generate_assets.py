"""
CatanMind Asset Generator
Auto-generates placeholder icon and splash images for APK build.
"""

import os
from pathlib import Path


def generate_placeholder_icon(output_path: str, size: int = 512):
    """Generate a placeholder app icon using matplotlib."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(size/100, size/100), dpi=100)
        ax.set_facecolor('#1a1a2e')
        
        # Draw hexagon
        angles = np.linspace(0, 2*np.pi, 7)
        x = 0.5 + 0.35 * np.cos(angles)
        y = 0.5 + 0.35 * np.sin(angles)
        ax.fill(x, y, color='#e94560', alpha=0.9)
        
        # Draw inner hexagon
        x_inner = 0.5 + 0.25 * np.cos(angles)
        y_inner = 0.5 + 0.25 * np.sin(angles)
        ax.fill(x_inner, y_inner, color='#16213e', alpha=1.0)
        
        # Add text
        ax.text(0.5, 0.5, 'CM', fontsize=size//10, fontweight='bold',
               ha='center', va='center', color='#e94560')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight', 
                   facecolor='#1a1a2e', edgecolor='none', pad_inches=0)
        plt.close()
        
        print(f"[OK] Generated icon: {output_path}")
        return True
        
    except ImportError as e:
        print(f"[WARN] matplotlib not available: {e}")
        return False


def generate_placeholder_splash(output_path: str, width: int = 1080, height: int = 1920):
    """Generate a placeholder splash screen."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import numpy as np
        
        fig, ax = plt.subplots(figsize=(width/100, height/100), dpi=100)
        ax.set_facecolor('#1a1a2e')
        
        # Draw large hexagon in center
        angles = np.linspace(0, 2*np.pi, 7)
        cx, cy = 0.5, 0.5
        x = cx + 0.2 * np.cos(angles)
        y = cy + 0.12 * np.sin(angles)
        ax.fill(x, y, color='#e94560', alpha=0.9)
        
        # Inner hexagon
        x_inner = cx + 0.14 * np.cos(angles)
        y_inner = cy + 0.08 * np.sin(angles)
        ax.fill(x_inner, y_inner, color='#16213e', alpha=1.0)
        
        # Text
        ax.text(0.5, 0.5, 'CM', fontsize=40, fontweight='bold',
               ha='center', va='center', color='#e94560')
        ax.text(0.5, 0.35, 'CatanMind', fontsize=24, fontweight='bold',
               ha='center', va='center', color='#eaeaea')
        ax.text(0.5, 0.30, 'AI Assistant', fontsize=14,
               ha='center', va='center', color='#888888')
        
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=100, bbox_inches='tight',
                   facecolor='#1a1a2e', edgecolor='none', pad_inches=0)
        plt.close()
        
        print(f"[OK] Generated splash: {output_path}")
        return True
        
    except ImportError as e:
        print(f"[WARN] matplotlib not available: {e}")
        return False


def main():
    """Generate all required assets."""
    base_dir = Path(__file__).parent.parent
    assets_dir = base_dir / "assets"
    
    print("Generating CatanMind assets...")
    print(f"Output directory: {assets_dir}")
    
    # Generate icon
    icon_path = assets_dir / "icon.png"
    generate_placeholder_icon(str(icon_path))
    
    # Generate splash
    splash_path = assets_dir / "splash.png"
    generate_placeholder_splash(str(splash_path))
    
    print("\nAsset generation complete!")


if __name__ == "__main__":
    main()
