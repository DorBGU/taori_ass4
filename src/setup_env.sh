	source /opt/ros/indigo/setup.bash
	source $HOME/catkin_ws/devel/setup.bash
	source /usr/share/gazebo/setup.sh
	export ROS_MASTER_URI="http://192.168.0.100:11311"
	echo $ROS_MASTER_URI
	export ROS_IP="192.168.0.102"  #my computer ip. ifconfig
	echo $ROS_IP
	export PS1="[ROS-ROBOT]"${PS1}