if (initialArray.length > 0) {
      const transformedData = initialArray.slice(0, initialArray.length - 1).map((value, index) => ({
        fileId: value,
        firmId: initialArray[index + 1]
      }));
      setRowData(transformedData);
    }
