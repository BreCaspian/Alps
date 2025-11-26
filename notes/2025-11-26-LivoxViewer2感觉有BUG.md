# 总感觉 LivoxViewer2 好像有BUG

<p align="center">
  <img src="../resource/figure/2025-11-26/Screenshot from 2025-11-26 20-22-38.png"
       alt="LivoxViewer 连接Livox Horizon激光雷达"
       height="400">
</p>
<p align="center"><em>图 1 LivoxViewer 连接 Livox Horizon 激光雷达 </em></p>


<p align="center">
  <img src="../resource/figure/2025-11-26/Screenshot from 2025-11-26 20-29-13.png"
       alt="livox_ros2_driver 驱动 Livox Horizon激光雷达"
       height="400">
</p>
<p align="center"><em>图 2 livox_ros2_driver 驱动 Livox Horizon 激光雷达 </em></p>

<p align="center">
  <img src="../resource/figure/2025-11-26/Screenshot from 2025-11-26 20-31-32.png"
       alt="LivoxViewer2 连接Livox Horizon激光雷达"
       height="400">
</p>
<p align="center"><em>图 3 LivoxViewer2 连接 Livox Horizon 激光雷达 </em></p>


不管是在Windows里面还是Linux里面，都有这种情况出现，激光雷达设备就是不给你显示出来，LivoxViewer2里面IP一直卡着第一次打开的IP

本身 Horizon 或着 PC 就要查看设备IP修改IP来连接，不给显示设备想修改都修改不了

反观LivoxViewer就没有这种情况，设备秒出，并且会显示设备IP，修改就很方便

还是LivoxViewer用其来顺手，驱动在修改IP后也很快连接上了
