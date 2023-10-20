# notepad
beforeFirstChunk: function(chunk) {
    // Replace any numeric values in scientific notation with their full representation
    return chunk.replace(/(\d+(\.\d+)?[Ee][+\-]?\d+)/g, function(match) {
      return `"${match}"`; // Wrap in double quotes to ensure it's treated as a string
    });
  },
