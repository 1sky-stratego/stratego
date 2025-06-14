#!/usr/bin/env python3
"""
Python Batch Runner
A simple script to run multiple Python files in sequence.
Place this in the same directory as your Python scripts.
"""

import os
import subprocess
import sys
import json
from pathlib import Path
from datetime import datetime

CONFIG_FILE = "batch_runner_config.json"

class PythonBatchRunner:
    def __init__(self):
        self.scripts = []
        self.config_file = Path(CONFIG_FILE)
        self.load_config()
    
    def load_config(self):
        """Load saved script list from config file"""
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    data = json.load(f)
                    self.scripts = data.get('scripts', [])
                print(f"✅ Loaded {len(self.scripts)} scripts from config")
            except Exception as e:
                print(f"⚠️  Error loading config: {e}")
                self.scripts = []
    
    def save_config(self):
        """Save current script list to config file"""
        try:
            config = {
                'scripts': self.scripts,
                'last_updated': datetime.now().isoformat()
            }
            with open(self.config_file, 'w') as f:
                json.dump(config, f, indent=2)
            print(f"💾 Config saved to {self.config_file}")
        except Exception as e:
            print(f"❌ Error saving config: {e}")
    
    def add_script(self, filename):
        """Add a Python script to the batch list"""
        # Clean up the filename
        filename = filename.strip()
        
        # Add .py extension if not present
        if not filename.endswith('.py'):
            filename += '.py'
        
        # Check if file exists
        if not Path(filename).exists():
            print(f"⚠️  Warning: {filename} not found in current directory")
            choice = input(f"Add anyway? (y/n): ").lower()
            if choice != 'y':
                return False
        
        # Check for duplicates
        if filename in self.scripts:
            print(f"⚠️  {filename} already in list")
            return False
        
        self.scripts.append(filename)
        print(f"✅ Added {filename} to batch list")
        return True
    
    def remove_script(self, filename_or_index):
        """Remove a script from the batch list"""
        try:
            # Try to remove by index first
            if filename_or_index.isdigit():
                index = int(filename_or_index) - 1  # Convert to 0-based index
                if 0 <= index < len(self.scripts):
                    removed = self.scripts.pop(index)
                    print(f"🗑️  Removed {removed} from batch list")
                    return True
                else:
                    print(f"❌ Invalid index. Use 1-{len(self.scripts)}")
                    return False
            
            # Try to remove by filename
            filename = filename_or_index.strip()
            if not filename.endswith('.py'):
                filename += '.py'
            
            if filename in self.scripts:
                self.scripts.remove(filename)
                print(f"🗑️  Removed {filename} from batch list")
                return True
            else:
                print(f"❌ {filename} not found in batch list")
                return False
                
        except Exception as e:
            print(f"❌ Error removing script: {e}")
            return False
    
    def list_scripts(self):
        """Display current script list"""
        if not self.scripts:
            print("📝 No scripts in batch list")
            return
        
        print(f"\n📋 Current Batch List ({len(self.scripts)} scripts):")
        print("-" * 50)
        for i, script in enumerate(self.scripts, 1):
            status = "✅" if Path(script).exists() else "❌"
            print(f"{i:2d}. {status} {script}")
        print("-" * 50)
    
    def run_all_scripts(self):
        """Run all scripts in the batch list"""
        if not self.scripts:
            print("📝 No scripts to run")
            return
        
        print(f"\n🚀 Starting batch execution of {len(self.scripts)} scripts...")
        print("=" * 60)
        
        results = []
        start_time = datetime.now()
        
        for i, script in enumerate(self.scripts, 1):
            print(f"\n[{i}/{len(self.scripts)}] Running: {script}")
            print("-" * 40)
            
            script_start = datetime.now()
            
            try:
                # Check if file exists
                if not Path(script).exists():
                    print(f"❌ File not found: {script}")
                    results.append({'script': script, 'status': 'FAILED', 'error': 'File not found'})
                    continue
                
                # Run the script
                result = subprocess.run(
                    [sys.executable, script],
                    capture_output=False,  # Show output in real-time
                    text=True,
                    cwd=os.getcwd()
                )
                
                script_duration = (datetime.now() - script_start).total_seconds()
                
                if result.returncode == 0:
                    print(f"✅ {script} completed successfully ({script_duration:.1f}s)")
                    results.append({'script': script, 'status': 'SUCCESS', 'duration': script_duration})
                else:
                    print(f"❌ {script} failed with exit code {result.returncode}")
                    results.append({'script': script, 'status': 'FAILED', 'exit_code': result.returncode})
                
            except KeyboardInterrupt:
                print(f"\n⚠️  Batch execution interrupted by user")
                results.append({'script': script, 'status': 'INTERRUPTED'})
                break
            except Exception as e:
                print(f"❌ Error running {script}: {e}")
                results.append({'script': script, 'status': 'ERROR', 'error': str(e)})
        
        # Summary
        total_duration = (datetime.now() - start_time).total_seconds()
        self.print_summary(results, total_duration)
    
    def print_summary(self, results, total_duration):
        """Print execution summary"""
        print(f"\n{'='*60}")
        print(f"📊 BATCH EXECUTION SUMMARY")
        print(f"{'='*60}")
        print(f"Total Duration: {total_duration:.1f} seconds")
        print(f"Scripts Processed: {len(results)}")
        
        success_count = sum(1 for r in results if r['status'] == 'SUCCESS')
        failed_count = len(results) - success_count
        
        print(f"✅ Successful: {success_count}")
        print(f"❌ Failed: {failed_count}")
        
        if results:
            print(f"\nDetailed Results:")
            for r in results:
                status_icon = "✅" if r['status'] == 'SUCCESS' else "❌"
                duration_str = f" ({r.get('duration', 0):.1f}s)" if 'duration' in r else ""
                error_str = f" - {r.get('error', r.get('exit_code', ''))}" if r['status'] != 'SUCCESS' else ""
                print(f"  {status_icon} {r['script']}{duration_str}{error_str}")
    
    def interactive_menu(self):
        """Main interactive menu"""
        while True:
            print(f"\n{'='*50}")
            print(f"🐍 PYTHON BATCH RUNNER")
            print(f"{'='*50}")
            self.list_scripts()
            
            print(f"\nOptions:")
            print(f"1. Add script")
            print(f"2. Remove script") 
            print(f"3. Run all scripts")
            print(f"4. Clear all scripts")
            print(f"5. Save and exit")
            print(f"6. Exit without saving")
            
            choice = input(f"\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                self.add_scripts_interactive()
            elif choice == '2':
                self.remove_script_interactive()
            elif choice == '3':
                self.run_all_scripts()
            elif choice == '4':
                self.clear_scripts()
            elif choice == '5':
                self.save_config()
                print("👋 Goodbye!")
                break
            elif choice == '6':
                print("👋 Goodbye! (Changes not saved)")
                break
            else:
                print("❌ Invalid choice. Please try again.")
    
    def add_scripts_interactive(self):
        """Interactive script addition"""
        print(f"\n➕ ADD SCRIPTS")
        print("Enter script names (without .py extension, or with)")
        print("Type 'done' when finished, 'list' to see .py files in current directory")
        
        while True:
            filename = input("Script name: ").strip()
            
            if filename.lower() == 'done':
                break
            elif filename.lower() == 'list':
                py_files = [f.name for f in Path('.').glob('*.py')]
                if py_files:
                    print("📁 Python files in current directory:")
                    for f in sorted(py_files):
                        print(f"  - {f}")
                else:
                    print("📁 No .py files found in current directory")
                continue
            elif filename == '':
                continue
            
            self.add_script(filename)
    
    def remove_script_interactive(self):
        """Interactive script removal"""
        if not self.scripts:
            print("📝 No scripts to remove")
            return
        
        print(f"\n🗑️  REMOVE SCRIPT")
        print("Enter script name or number to remove:")
        
        choice = input("Remove: ").strip()
        if choice:
            self.remove_script(choice)
    
    def clear_scripts(self):
        """Clear all scripts from the list"""
        if not self.scripts:
            print("📝 No scripts to clear")
            return
        
        confirm = input(f"🗑️  Clear all {len(self.scripts)} scripts? (y/n): ").lower()
        if confirm == 'y':
            self.scripts.clear()
            print("🗑️  All scripts cleared")

def main():
    """Main entry point"""
    runner = PythonBatchRunner()
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == 'run':
            # Direct run mode
            runner.run_all_scripts()
        elif sys.argv[1] == 'list':
            # List scripts mode
            runner.list_scripts()
        elif sys.argv[1] == 'add':
            # Add scripts from command line
            for script in sys.argv[2:]:
                runner.add_script(script)
            runner.save_config()
        else:
            print("Usage: python batch_runner.py [run|list|add script1 script2 ...]")
    else:
        # Interactive mode
        runner.interactive_menu()

if __name__ == "__main__":
    main()