#!/usr/bin/env bash

echo "Installing necessary packages for student ...."

# Install from requirements.txt
read -r -p "Do you want uninstall as root? [Y/n] " input

case $input in
    [yY][eE][sS]|[yY])
        echo "Installing as root . . . "
        sudo su
        pip install -r requirements-students-human2d.txt
        mkdir human_2d
        cd human_2d
        git clone https://gitlab.com/BioRobAnimals/Human_2D/webots.git
        git clone https://gitlab.com/BioRobAnimals/Human_2D/modeling.git
        ;;
    [nN][oO]|[nN])
        echo "Installing as user . . . "
        pip install --user -r requirement_copy.txt
        mkdir human_2d
        cd human_2d
        git clone https://gitlab.com/BioRobAnimals/Human_2D/webots.git
        git clone https://gitlab.com/BioRobAnimals/Human_2D/modeling.git
        ;;
    *)
        echo "Invalid input..."
        exit 1
        ;;
esac
