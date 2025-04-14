import itk

# Assume InputPixelType and OutputPixelType are defined, for example:
InputPixelType = itk.F
OutputPixelType = itk.UC  # typically unsigned char for 0-255 intensity range

# Define image types
InputImageType = itk.Image[InputPixelType, 3]  # Replace 2 with image dimension if needed
OutputImageType = itk.Image[OutputPixelType, 3]

def rescale_intensity(itk_image):
    rescale_filter = itk.RescaleIntensityImageFilter[InputImageType, OutputImageType].New()
    rescale_filter.SetInput(itk_image)
    rescale_filter.SetOutputMinimum(0)
    rescale_filter.SetOutputMaximum(255)

    # Run the filter
    rescale_filter.Update()
    output_itk_image = rescale_filter.GetOutput()

    return output_itk_image
