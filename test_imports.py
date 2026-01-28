#!/usr/bin/env python3
"""
Simple test script to verify the refactored pipeline modules can be imported.
Run this to check that all imports work correctly before running the full pipeline.
"""

def test_imports():
    """Test that all pipeline modules can be imported."""
    print("Testing imports...")
    
    try:
        print("  ✓ Importing pipeline.utils...")
        from pipeline.utils import resolve_device, cls_to_yards
        print("    - resolve_device: OK")
        print("    - cls_to_yards: OK")
        
        print("  ✓ Importing pipeline.models...")
        from pipeline.models import load_models
        print("    - load_models: OK")
        
        print("  ✓ Importing pipeline.detection...")
        from pipeline.detection import detect_hash_and_numbers, detect_players_and_poses
        print("    - detect_hash_and_numbers: OK")
        print("    - detect_players_and_poses: OK")
        
        print("  ✓ Importing pipeline.processing...")
        from pipeline.processing import fuse_detections
        print("    - fuse_detections: OK")
        
        print("  ✓ Importing pipeline.geometry...")
        from pipeline.geometry import estimate_homography, homography_quality, smooth_homography
        print("    - estimate_homography: OK")
        print("    - homography_quality: OK")
        print("    - smooth_homography: OK")
        
        print("  ✓ Importing pipeline.visualization...")
        from pipeline.visualization import draw_overlay
        print("    - draw_overlay: OK")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {e}")
        return False


def test_basic_functions():
    """Test basic functionality of utility functions."""
    print("\nTesting basic functions...")
    
    try:
        from pipeline.utils import resolve_device, cls_to_yards
        
        # Test resolve_device
        print("  ✓ Testing resolve_device...")
        device = resolve_device("auto")
        print(f"    - Auto-detected device: {device}")
        assert device in ["cpu", "mps", "0"], f"Unexpected device: {device}"
        
        # Test cls_to_yards
        print("  ✓ Testing cls_to_yards...")
        assert cls_to_yards(0) == 10, "cls_to_yards(0) should return 10"
        assert cls_to_yards(1) == 20, "cls_to_yards(1) should return 20"
        assert cls_to_yards(2) == 30, "cls_to_yards(2) should return 30"
        assert cls_to_yards(3) == 40, "cls_to_yards(3) should return 40"
        assert cls_to_yards(4) == 50, "cls_to_yards(4) should return 50"
        assert cls_to_yards(99) is None, "cls_to_yards(99) should return None"
        print("    - All mappings correct")
        
        print("\n✅ All basic tests passed!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=" * 60)
    print("Pipeline Module Import Test")
    print("=" * 60)
    print()
    
    import_success = test_imports()
    
    if import_success:
        test_success = test_basic_functions()
    else:
        test_success = False
    
    print()
    print("=" * 60)
    if import_success and test_success:
        print("✅ All tests passed! The refactored pipeline is ready to use.")
    else:
        print("❌ Some tests failed. Please check the errors above.")
    print("=" * 60)


if __name__ == "__main__":
    main()
