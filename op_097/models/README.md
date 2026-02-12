## Supercombo
### Supercombo input format (Full size: 799906 x float32)
* **image stream**
  * Two consecutive images (256 * 512 * 3 in RGB) recorded at 20 Hz : 393216 = 2 * 6 * 128 * 256
    * Each 256 * 512 image is represented in YUV420 with 6 channels : 6 * 128 * 256
      * Channels 0,1,2,3 represent the full-res Y channel and are represented in numpy as Y[::2, ::2], Y[::2, 1::2], Y[1::2, ::2], and Y[1::2, 1::2]
      * Channel 4 represents the half-res U channel
      * Channel 5 represents the half-res V channel
* **wide image stream**
  * Two consecutive images (256 * 512 * 3 in RGB) recorded at 20 Hz : 393216 = 2 * 6 * 128 * 256
    * Each 256 * 512 image is represented in YUV420 with 6 channels : 6 * 128 * 256
      * Channels 0,1,2,3 represent the full-res Y channel and are represented in numpy as Y[::2, ::2], Y[::2, 1::2], Y[1::2, ::2], and Y[1::2, 1::2]
      * Channel 4 represents the half-res U channel
      * Channel 5 represents the half-res V channel
* **desire**
  * one-hot encoded buffer to command model to execute certain actions, bit needs to be sent for the past 5 seconds (at 20FPS) : 100 * 8
* **traffic convention**
  * one-hot encoded vector to tell model whether traffic is right-hand or left-hand traffic : 2
* **feature buffer**
  * A buffer of intermediate features that gets appended to the current feature to form a 5 seconds temporal context (at 20FPS) : 99 * 512


### Supercombo output format (Full size: XXX x float32)
Read [here](https://github.com/commaai/openpilot/blob/90af436a121164a51da9fa48d093c29f738adf6a/selfdrive/modeld/models/driving.h#L236) for more.
