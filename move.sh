#!/bin/bash

# Define source and destination directories
source_dir="$1"
dest_dir="$2"

# Check if arguments are provided
if [ -z "$source_dir" ] || [ -z "$dest_dir" ]; then
  echo "Usage: $0 <source_dir> <destination_dir>"
  exit 1
fi

# Check if source directory exists
if [ ! -d "$source_dir" ]; then
  echo "Error: Source directory '$source_dir' does not exist."
  exit 1
fi

# Check if destination directory exists (and create it if not)
if [ ! -d "$dest_dir" ]; then
  mkdir -p "$dest_dir"
fi

# Loop through files in the source directory
for file in "$source_dir/"*; do
  # Skip directories
  if [ -d "$file" ]; then
    continue
  fi

  # Check if file already exists in destination directory
  if [ -f "$dest_dir/$file" ]; then
    echo "Skipping '$file' (already exists)"
  else
    # Move the file
    mv "$file" "$dest_dir"
    echo "Moved '$file' to destination directory"
  fi
done

echo "Finished moving files."
