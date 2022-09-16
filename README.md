# Autonomous Vehicle Lane Change Decision Making with Nash Q-Learning
In order to ensure a safer and more reliable path during the lane change process, the motion decision algorithm needs to predict the possibility of different interaction behaviors of the surrounding vehicles and then make a beneficial decision based on that. For this purpose, a motion decision-making method is proposed considering the interaction of surrounding vehicles. An interactive motion prediction method based on game theory is developed to predict the possibility of interactive behaviors and future local trajectories of surrounding vehicles. A motion decision algorithm based on Nash Q learning is implemented for an autonomous vehicle. Finally, the motion decision algorithm is validated in the lane change scene and compared with the rule-based lane change decision algorithm. The results show that this study's decision-making method is superior in terms of safety and efficiency and can effectively predict the interaction of nearby vehicles.

<img src="https://github.com/alireza-montazeri/AV-Nash-Q-Learning/blob/master/figures/Scenario.png" />

The self-driving car (blue) is looking for an opportunity to change to the left lane. It is necessary to anticipate the interaction of the surrounding vehicle (yellow) that is behind in the target line, such as slowing down, maintaining, or accelerating. The first simulation mode is that the style of the yellow car is gentle, which means that when the blue car changes direction, the yellow car tends to slow down. The second simulation mode is that the style of the yellow vehicle is aggressive, meaning that the yellow vehicle tends to accelerate when the blue vehicle is about to change lanes. In these two scenarios, the blue vehicle initially does not recognize the driving style of the yellow vehicle for the target lane. Therefore, he will try to judge the driving style of the yellow car by changing the lane and observing the reaction of the yellow car. The simulation is done in a python environment. The highway-env library is used to implement the environment. The learning process is performed for 100,000 repetitions, which lasts for 68 minutes.
## Result
### Gentle surrounding vehicle
In the First scenario, the driving style of the yellow vehicle is gentle, which means that the yellow vehicle will cooperate with the autonomous vehicle after it detects a desire to change lanes. Due to the two-to-one lane merging, the self-driving car should choose an opportunity to change lanes, which in addition to improving the driving reward, will enjoy high driving safety. A self-driving car that adopts a rule-based decision-making algorithm considers surrounding vehicles as either a constant speed obstacle or a uniform acceleration obstacle, regardless of interaction. However, the Nash Q-Learning based motion decision algorithm can choose a safe time of 1 second to change lanes and predict the most likely action of the interactive vehicle by playing the game with the yellow vehicle.
Position           |  Velocity
:-------------------------:|:-------------------------:
<img src="https://github.com/alireza-montazeri/AV-Nash-Q-Learning/blob/master/figures/gentle_position.png" /> | <img src="https://github.com/alireza-montazeri/AV-Nash-Q-Learning/blob/master/figures/gentle_velocity.png"  />

### Aggressive surrounding vehicle
In this scenario, the yellow vehicle's driving style is aggressive, meaning that the yellow vehicle accelerates to occupy the lane after detecting the autonomous vehicle's desire to change lanes. As seen in Figure, the self-driving car that adopts the rule-based decision algorithm starts changing lanes at t=0.5 seconds. However, suppose the yellow vehicle does not slow down. In that case, the ego car will collide with the yellow vehicle in the intended lane within 1.5 seconds, leading to a high risk for the autonomous vehicle to change lanes. On the other hand, the self-driving car with the Nash Q-Learning algorithm predicts that the interactive behavior of the yellow car during the game is to accelerate, and at first, it returns to the initial line and slows down. Finally, at t=3 seconds, he changes the lane again.
Position           |  Velocity
:-------------------------:|:-------------------------:
<img src="https://github.com/alireza-montazeri/AV-Nash-Q-Learning/blob/master/figures/aggressive_position.png" />|<img src="https://github.com/alireza-montazeri/AV-Nash-Q-Learning/blob/master/figures/aggressive_velocity.png" />

### Simulation

SV Type     |   Nash Q-Learning           |  Rule Based
:-------------------------:|:-------------------------:|:-------------------------:
Gentle | <img src="https://github.com/alireza-montazeri/AV-Nash-Q-Learning/blob/master/movies/nashQ_gentle.gif" />|<img src="https://github.com/alireza-montazeri/AV-Nash-Q-Learning/blob/master/movies/rule_gentle.gif" />
Aggressive | <img src="https://github.com/alireza-montazeri/AV-Nash-Q-Learning/blob/master/movies/nashQ_aggressive.gif" />|<img src="https://github.com/alireza-montazeri/AV-Nash-Q-Learning/blob/master/movies/rule_aggressive.gif" />

## References
[1]	C. Xu, W. Zhao, L. Li, Q. Chen, D. Kuang and J. Zhou, "A Nash Q-Learning Based Motion Decision Algorithm With Considering Interaction to Traffic Participants," in IEEE Transactions on Vehicular Technology, vol. 69, no. 11, pp. 12621-12634, Nov. 2020, doi: 10.1109/TVT.2020.3027352.

[2]	Junling Hu and Michael P. Wellman. 2003. Nash q-learning for general-sum stochastic games. J. 	Mach. Learn. Res 4, (12/1/2003), 1039–1069.

[3]	Yang, D., Jin, J.P., Pu, Y., et al.: ‘Safe distance car-following model including backward-looking and its stability analysis’, Eur. Phys. J. B, 2013, 86, p. 92.

[4]	Leurent, E. (2018). An Environment for Autonomous Driving Decision-Making. In GitHub repository. GitHub. https://github.com/eleurent/highway-env

[5]	Zhou, X., Kuang, D., Zhao, W., Xu, C., Feng, J. and Wang, C. (2020), Lane-changing decision method based Nash Q-learning with considering the interaction of surrounding vehicles. IET Intell. Transp. Syst., 14: 2064-2072
