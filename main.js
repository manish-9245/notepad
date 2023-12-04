  const isValidFirmId = (value) => firmIdOptions.some((option) => option.value === value);

  // Transform data to desired format
  const rowData = initialArray.reduce((acc, item, index) => {
    // Check if current item and next item are both defined
    if (item && initialArray[index + 1]) {
      acc.push({ fileId: item, firmId: isValidFirmId(initialArray[index + 1]) ? initialArray[index + 1] : '' });
    } else if (item) {
      // Handle cases with single element or last element
      acc.push({ fileId: item, firmId: '' });
    }
    return acc;
  }
