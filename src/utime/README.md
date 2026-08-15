# U-Time

This module implements artifact detection using the U-Time architecture developed by Mathias Perslev et al. 

**Original Repository:** [https://github.com/perslev/U-Time](https://github.com/perslev/U-Time)
**Paper:** [U-Time: A Fully Convolutional Network for Time Series Segmentation Applied to Sleep Staging](http://papers.nips.cc/paper/8692-u-time-a-fully-convolutional-network-for-time-series-segmentation-applied-to-sleep-staging.pdf)

While the original author intended this for sleep segmentation, I have repurposed the architecture for segmentation of chewing artifacts present in our data. These artifacts were quite obvious and not so difficult to mark manually. For this size of dataset, the time it took to implement this was roughly equal to what it would have taken to label all data by hand, but I'll appreciate it if I ever run into a similar problem on a bigger dataset.

The code definitely needs some polishing and better organization. That should be possible to do from this saved version, if I ever have a need it again.
