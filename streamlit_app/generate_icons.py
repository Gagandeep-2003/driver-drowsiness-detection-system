#!/usr/bin/env python3
"""
PWA Icon Generator for Driver Drowsiness Detection System
Generates all required PWA icons from a base design
"""

import os
from PIL import Image, ImageDraw, ImageFont
import sys

def create_icon(size, output_path):
    """Create a PWA icon with the specified size"""
    # Create a new image with a gradient background
    img = Image.new('RGBA', (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    
    # Create a gradient background
    for i in range(size):
        alpha = int(255 * (1 - i / size))
        color = (255, 107, 107, alpha)  # Red gradient matching theme
        draw.ellipse([i//4, i//4, size-i//4, size-i//4], fill=color)
    
    # Add a car/eye icon in the center
    center = size // 2
    icon_size = size // 3
    
    # Draw a stylized eye
    eye_width = icon_size
    eye_height = icon_size // 2
    eye_x = center - eye_width // 2
    eye_y = center - eye_height // 2
    
    # Outer eye shape
    draw.ellipse([eye_x, eye_y, eye_x + eye_width, eye_y + eye_height], 
                fill='white', outline='black', width=2)
    
    # Inner pupil
    pupil_size = eye_height // 2
    pupil_x = center - pupil_size // 2
    pupil_y = center - pupil_size // 2
    draw.ellipse([pupil_x, pupil_y, pupil_x + pupil_size, pupil_y + pupil_size], 
                fill='black')
    
    # Add text if icon is large enough
    if size >= 128:
        try:
            # Try to use a nice font, fallback to default
            font_size = max(12, size // 20)
            font = ImageFont.load_default()
            
            text = "DDD"
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            
            text_x = center - text_width // 2
            text_y = center + eye_height // 2 + 10
            
            draw.text((text_x, text_y), text, fill='white', font=font)
        except:
            pass
    
    # Save the icon
    img.save(output_path, 'PNG')
    print(f"Created icon: {output_path} ({size}x{size})")

def generate_all_icons():
    """Generate all required PWA icons"""
    sizes = [72, 96, 128, 144, 152, 192, 384, 512]
    icons_dir = os.path.join(os.path.dirname(__file__), 'icons')
    
    # Create icons directory if it doesn't exist
    os.makedirs(icons_dir, exist_ok=True)
    
    for size in sizes:
        icon_path = os.path.join(icons_dir, f'icon-{size}x{size}.png')
        create_icon(size, icon_path)
    
    print(f"\n✅ Generated {len(sizes)} PWA icons in {icons_dir}")
    print("Icons are ready for PWA installation!")

if __name__ == "__main__":
    try:
        generate_all_icons()
    except ImportError as e:
        print("❌ Error: PIL (Pillow) is required to generate icons.")
        print("Install it with: pip install Pillow")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error generating icons: {e}")
        sys.exit(1)
