1. git clone
2. install graphviz

   a) For Windows:

      i. Download Graphviz: Visit the [Graphviz website](https://graphviz.org/download/) and download the installer for Windows.

      ii. Run the Installer: Once downloaded, run the installer and follow the installation instructions.

      iii. Set Path (Optional): During installation, you may have the option to add Graphviz to your system's PATH environment variable.
          If not, you can manually add it later by going to Control Panel > System and Security > System > Advanced system settings > Environment Variables,
          then editing the PATH variable to include the Graphviz bin directory (e.g., C:\Program Files\Graphviz\bin).

   b) For macOS:

      i. Using Homebrew: brew install graphviz

      ii. Using MacPorts: sudo port install graphviz

   c) For Linux (Ubuntu/Debian):

      i. Using Apt: sudo apt-get update
                   sudo apt-get install graphviz

      ii. Using Yum (CentOS/RHEL): sudo yum install graphviz

      iii. Using Zypper (openSUSE): sudo zypper install graphviz

4. cargo run

  

