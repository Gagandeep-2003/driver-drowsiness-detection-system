#!/usr/bin/env python3
"""
PWA Setup Verification Script
Tests if all PWA components are properly configured
"""

import os
import json
import sys
from pathlib import Path

def check_file_exists(file_path, description):
    """Check if a file exists and print status"""
    if os.path.exists(file_path):
        print(f"✅ {description}: {file_path}")
        return True
    else:
        print(f"❌ {description}: {file_path} (MISSING)")
        return False

def validate_manifest(manifest_path):
    """Validate the PWA manifest file"""
    try:
        with open(manifest_path, 'r') as f:
            manifest = json.load(f)
        
        required_fields = ['name', 'short_name', 'start_url', 'display', 'icons']
        missing_fields = []
        
        for field in required_fields:
            if field not in manifest:
                missing_fields.append(field)
        
        if missing_fields:
            print(f"❌ Manifest missing required fields: {missing_fields}")
            return False
        else:
            print(f"✅ Manifest validation passed")
            print(f"   📱 App Name: {manifest['name']}")
            print(f"   🏷️  Short Name: {manifest['short_name']}")
            print(f"   🎨 Theme Color: {manifest.get('theme_color', 'Not set')}")
            print(f"   🖼️  Icons: {len(manifest['icons'])} sizes")
            return True
            
    except json.JSONDecodeError:
        print(f"❌ Manifest JSON is invalid")
        return False
    except FileNotFoundError:
        print(f"❌ Manifest file not found")
        return False

def check_icons(icons_dir):
    """Check if all required PWA icons exist"""
    required_sizes = ['72x72', '96x96', '128x128', '144x144', '152x152', '192x192', '384x384', '512x512']
    missing_icons = []
    
    for size in required_sizes:
        icon_path = os.path.join(icons_dir, f'icon-{size}.png')
        if not os.path.exists(icon_path):
            missing_icons.append(f'icon-{size}.png')
    
    if missing_icons:
        print(f"❌ Missing icons: {missing_icons}")
        return False
    else:
        print(f"✅ All {len(required_sizes)} PWA icons present")
        return True

def main():
    """Main verification function"""
    print("🔍 PWA Setup Verification")
    print("=" * 50)
    
    base_dir = os.path.dirname(__file__)
    
    # Check PWA files
    files_to_check = [
        (os.path.join(base_dir, 'manifest.json'), 'PWA Manifest'),
        (os.path.join(base_dir, 'sw.js'), 'Service Worker'),
        (os.path.join(base_dir, 'offline.html'), 'Offline Page'),
        (os.path.join(base_dir, 'streamlit_app_pwa.py'), 'Enhanced PWA App'),
        (os.path.join(base_dir, '.streamlit', 'config.toml'), 'Streamlit Config'),
    ]
    
    all_files_exist = True
    for file_path, description in files_to_check:
        if not check_file_exists(file_path, description):
            all_files_exist = False
    
    print("\n📱 PWA Manifest Validation")
    print("-" * 30)
    manifest_valid = validate_manifest(os.path.join(base_dir, 'manifest.json'))
    
    print("\n🖼️ PWA Icons Check")
    print("-" * 20)
    icons_dir = os.path.join(base_dir, 'icons')
    icons_valid = check_icons(icons_dir)
    
    print("\n📊 Overall PWA Status")
    print("=" * 25)
    
    if all_files_exist and manifest_valid and icons_valid:
        print("🎉 PWA Setup Complete!")
        print("\nNext steps:")
        print("1. Run: streamlit run streamlit_app_pwa.py")
        print("2. Open in browser with HTTPS")
        print("3. Look for install prompt or button")
        print("4. Test offline functionality")
        return True
    else:
        print("❌ PWA Setup Incomplete")
        print("\nTo fix issues:")
        if not all_files_exist:
            print("- Ensure all PWA files are present")
        if not manifest_valid:
            print("- Fix manifest.json validation errors")
        if not icons_valid:
            print("- Generate missing icons: python3 generate_icons.py")
        return False

def test_pwa_features():
    """Test PWA-specific features"""
    print("\n🧪 Testing PWA Features")
    print("-" * 25)
    
    # Test icon generation
    try:
        from PIL import Image
        print("✅ PIL/Pillow available for icon generation")
    except ImportError:
        print("❌ PIL/Pillow not available - install with: pip install Pillow")
    
    # Test Streamlit dependencies
    try:
        import streamlit
        print(f"✅ Streamlit {streamlit.__version__} available")
    except ImportError:
        print("❌ Streamlit not available")
    
    # Test if running in HTTPS context (for production)
    print("💡 Note: PWA installation requires HTTPS in production")

if __name__ == "__main__":
    success = main()
    test_pwa_features()
    
    if success:
        print("\n🚀 Ready to launch PWA!")
        sys.exit(0)
    else:
        print("\n🔧 Please fix the issues above before launching")
        sys.exit(1)
