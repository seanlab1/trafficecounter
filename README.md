# OpenCV-Traffic-Counter
This project details how to create a simple traffic counter designed using the OpenCV library for Python 3.5, and was originally carried out as part of the [Government Data Science Accelerator programme](https://gdsdata.blog.gov.uk/2017/08/11/pharmacies-people-and-ports-the-data-science-accelerator/) in June-October 2017.
오픈CV-트래픽 카운터
이 프로젝트는 Python 3.5용 OpenCV 라이브러리를 사용하여 설계된 간단한 트래픽 카운터를 만드는 방법을 자세히 설명하며 원래 2017년 6월-10월에 정부 데이터 과학 가속기 프로그램의 일부로 수행되었습니다.

<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/outputs/outputScreen.png?raw=true" width="352">

## The Project
The purpose of this project was to detect and count vehicles from a CCTV feed, and to ultimately give an idea of what the real-time on street situation is across the road network (in this case within Greater London). To that end, the TfL JamCam API was used throughout to test the algorithm. This is an API provided by Transport for London and can be used to obtain ~10 second clips of road traffic across the London road network, an example of which can be seen [here](https://s3-eu-west-1.amazonaws.com/jamcams.tfl.gov.uk/00002.00625.mp4).
이 프로젝트의 목적은 CCTV 피드에서 차량을 감지하고 계산하고 궁극적으로 도로망 (이 경우 그레이터 런던 내)에서 실시간 거리 상황이 무엇인지에 대한 아이디어를 제공하는 것이 었습니다. 이를 위해 TfL JamCam API를 사용하여 알고리즘을 테스트했습니다. 이것은 Transport for London에서 제공하는 API이며 런던 도로망을 가로지르는 도로 교통의 ~10초 클립을 얻는 데 사용할 수 있으며 그 예는 여기에서 볼 수 있습니다.
The main code can be found in [/trafficCounter/blobDetection.py](https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/blobDetection.py) along with some other useful scripts that will assist with extracting [individual frames](https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/frame_extract.py), [histograms](https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/histogram_extraction.py) for illustrating how different conditions affect each frame, and [/trafficCounter/createSeedFiles.py](https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/createSeedFiles.py) and [/trafficCounter/haarCascades.py](https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/haarCascades.py) for starting work with HAAR cascades as an alternative method to blob detection (work in progress).
기본 코드는 /trafficCounter/blobDetection.py 개별 프레임 추출에 도움이 되는 몇 가지 유용한 스크립트, 다양한 조건이 각 프레임에 미치는 영향을 보여 주는 히스토그램, /trafficCounter/createSeedFiles.py 및 /trafficCounter/haarCascades에서 찾을 수 있습니다.py blob 감지(진행 중인 작업)의 대체 방법으로 HAAR 캐스케이드로 작업을 시작합니다.

## Method
### Object Detection
In order to count vehicles we first need to be able to detect them in an image. This is pretty simple for a human to pick out but harder to implement in the machine world. However, if we consider that an image is just an array of numbers (one value per pixel), we may be able to use this to determine what a vehicle looks like and what we'd expect to see when there isn't a vehicle there. We can use OpenCV to look at how the value of certain pixels changes for these two conditions, as shown in the image below. To do this, we must first translate our image from RGB channels (Red, Green Blue) to HSV (Hue, Saturation, Value) and inspect each channel to see if it can tell us something.
차량을 계산하려면 먼저 이미지에서 차량을 감지 할 수 있어야합니다. 이것은 인간이 선택하기에는 매우 간단하지만 기계 세계에서는 구현하기가 더 어렵습니다. 그러나 이미지가 숫자의 배열(픽셀당 하나의 값)일 뿐이라고 생각하면 이를 사용하여 차량이 어떻게 생겼는지, 차량이 없을 때 무엇을 볼 수 있는지 확인할 수 있습니다.
OpenCV를 사용하여 아래 이미지와 같이이 두 조건에 대해 특정 픽셀의 값이 어떻게 변경되는지 확인할 수 있습니다. 이렇게하려면 먼저 RGB 채널 (빨강, 녹색 파랑)에서 HSV (색조, 채도, 값)로 이미지를 변환하고 각 채널을 검사하여 무언가를 알려줄 수 있는지 확인해야합니다.

<br>
<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/outputs/carNoCar.png?raw=true" width="400"><br>
<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/outputs/hsv.png?raw=true" width="400"><br>
As we can see from the histogram plots, the Hue channel does not offer much information, whereas both the Saturation and Value channels clearly show a difference between the Vehicle/No Vehicle conditions and so we can use this channels in our detection algorithm. However, for simplicity we will just use the Value channel for the time being.
히스토그램 플롯에서 볼 수 있듯이 색조 채널은 많은 정보를 제공하지 않는 반면, 채도 및 값 채널 모두 차량/차량 없음 조건 간의 차이를 명확하게 보여주므로 감지 알고리즘에서 이 채널을 사용할 수 있습니다. 그러나 단순화를 위해 당분간은 Value 채널만 사용합니다.
We can then use this information to determine what is background and what is a vehicle, so long as we have a suitable background image ie a version of our scene with no vehicles in it. In the case shown here it is very difficult to obtain a clear image, however we can use OpenCV to average between several frames and create our background image.
그런 다음 이 정보를 사용하여 적절한 배경 이미지(예: 차량이 없는 장면 버전)가 있는 한 배경과 차량이 무엇인지 결정할 수 있습니다. 여기에 표시된 경우 선명한 이미지를 얻는 것은 매우 어렵지만 OpenCV를 사용하여 여러 프레임 사이의 평균을 내고 배경 이미지를 만들 수 있습니다.

<br>
<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/backgrounds/625_bg.jpg?raw=true" width="352"><br>
Now that we have a background image, or an array of default/background values, we can use OpenCV to detect when these values go above a certain value (or 'threshold value'). We assume that this occurs when there is a vehicle within that pixel, and so use OpenCV to set the pixels that meet the threshold criteria to maximum brightness (this will make detecting shapes/vehicles easier later on).
이제 배경 이미지 또는 기본값 / 배경 값 배열이 있으므로 OpenCV를 사용하여 이러한 값이 특정 값 (또는 '임계 값')을 초과하는지 감지 할 수 있습니다. 해당 픽셀 내에 차량이있을 때 발생한다고 가정하므로 OpenCV를 사용하여 임계 값 기준을 충족하는 픽셀을 최대 밝기로 설정합니다 (나중에 모양 / 차량을 더 쉽게 감지 할 수 있음).
<br>
<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/outputs/thresh.png?raw=true" width="352">
<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/outputs/blobs.png?raw=true" width="352"><br>
The images above show the pixels that meet the threshold criteria (left) and the resulting shapes after setting those pixels to maximum value/brightness (right). Also highlighted (green) is gaps in our objects where dark areas (windscreens, grills etc) may not meet our threshold criteria. This could cause a problem later on so we try to fill in these gaps using the erosion and dilation functions from the OpenCV library.
위의 이미지는 임계값 기준을 충족하는 픽셀(왼쪽)과 해당 픽셀을 최대값/밝기(오른쪽)로 설정한 후의 결과 모양을 보여줍니다. 또한 강조 표시(녹색)는 어두운 영역(앞유리, 그릴 등)이 임계값 기준을 충족하지 않을 수 있는 물체의 간격입니다. 이것은 나중에 문제를 일으킬 수 있으므로 OpenCV 라이브러리의 침식 및 팽창 함수를 사용하여 이러한 간격을 채우려고합니다.
Once we are happy with the shapes created, we must then check the shapes (or contours) to determine which are most like to be vehicles before dismissing those that are not. We can do this be implementing a condition where we are only interested in the detected contours if they are over a certain size. Note that this will change depending on the video feed. The kept contours can then be passed to the [Vehicle Counter algorithm](https://stackoverflow.com/a/36274515), based on the one created by Dan Maesk.
생성된 모양에 만족하면 모양(또는 윤곽선)을 확인하여 차량이 아닌 모양을 닫기 전에 차량과 가장 유사한 모양을 결정해야 합니다. 감지된 윤곽선이 특정 크기를 초과하는 경우에만 관심이 있는 조건을 구현할 수 있습니다. 이는 비디오 피드에 따라 변경됩니다. 그런 다음 유지된 윤곽을 [차량 카운터 알고리즘]에 전달할 수 있습니다.(https://stackoverflow.com/a/36274515), based on the one created by Dan Maesk(https://stackoverflow.com/a/36274515), based on the one created by Dan Maesk

### Counting Vehicles
The vehicle counter is split into two class objects, one named `Vehicle` which is used to define each vehicle object, and the other `Vehicle Counter` which determines which 'vehicles' are valid before counting them (or not). `Vehicle` is relatively simple and offers information about each detected object such as a tracked position in each frame, how many frames it has appeared in (and how many it has not been seen for if we temporarily loose track of it), whether we have counted the vehicle yet and what direction we believe the vehicle to be travelling in. We can also obtain the last position and the position before that in order to calculate a few values within our `Vehicle Counter` algorithm.
차량 카운터는 두 개의 클래스 객체로 나뉘는데, 하나는 각 차량 객체를 정의하는 데 사용되는 Vehicle이라는 이름의 객체이고, 다른 하나는 어떤 '차량'이 유효한지 계산하기 전에 유효한지 여부를 결정하는 Vehicle Counter입니다. 차량은 비교적 간단하며 각 프레임의 추적 위치, 얼마나 많은 프레임에 나타 났는지 (그리고 일시적으로 추적을 느슨하게하면 몇 개의 프레임이 보이지 않았는지)와 같은 감지 된 각 물체에 대한 정보를 제공합니다.
차량을 아직 계산했는지 여부와 차량이 주행할 것으로 생각되는 방향. 차량 카운터 알고리즘 내에서 몇 가지 값을 계산하기 위해 마지막 위치와 그 이전의 위치를 얻을 수도 있습니다.

`Vehicle Counter` is more complex and serves several purposes. We can use it to determine the vector movement of each tracked vehicle from frame to frame, giving an indicator of what movements are true and which are false matches. We do this to make sure we're not incorrectly matching vehicles and therefore getting the most accurate count possible. In this case, we only expect vehicles travelling from the top of the image to the bottom right hand corner, or the reverse. This means we only have a certain range of allowable vector movements based on the angle that the vehicle has moved - this can be seen from the images below. The image on the left shows the expected vector movements (highlighted in red) and the image on the left shows a chart of distance moved vs the angle - those classed as allowable movements are highlighted by the green box.
차량 카운터는 더 복잡하며 여러 가지 용도로 사용됩니다. 이를 사용하여 프레임에서 프레임으로 추적된 각 차량의 벡터 움직임을 결정하여 어떤 움직임이 참이고 어떤 움직임이 거짓 일치인지에 대한 지표를 제공할 수 있습니다. 이렇게 하면 차량이 잘못 일치하지 않아 가능한 가장 정확한 개수를 얻을 수 있습니다.
이 경우 이미지 상단에서 오른쪽 하단 모서리로 또는 그 반대로 이동하는 차량만 예상합니다. 이것은 차량이 움직인 각도에 따라 허용 가능한 벡터 이동의 특정 범위만 갖는다는 것을 의미하며, 이는 아래 이미지에서 볼 수 있습니다. 왼쪽 이미지는 예상되는 벡터 이동(빨간색으로 강조 표시됨)을 보여주고 왼쪽 이미지는 이동한 거리 대 각도 차트를 보여줍니다.
-허용되는 움직임으로 분류 된 것은 녹색 상자로 강조 표시됩니다.

<br>
<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/outputs/vector.png?raw=true" width="175">
<img src="https://github.com/alex-drake/OpenCV-Traffic-Counter/blob/master/trafficCounter/outputs/vectorMovements.png?raw=true" width="400"><br>
**Note that this section of code is ripe for improvements as the change in angle is likely a better indicator of a true match than absolute angle but this has not yet been implemented.**
**각도의 변화가 절대 각도보다 실제 일치의 더 나은 지표일 가능성이 높지만 아직 구현되지 않았기 때문에 이 코드 섹션은 개선이 무르익었습니다.**

If a vehicle object satisfies the above criteria, we then want to check what direction it is moving in before then passing it to the counter. We can then use this information to determine whether the vehicle should be counted and then whether the count applies to the left hand lanes (up direction) or right hand lanes (down direction). Once satisfied, we update the counter and print it to the output frame. If a vehicle has not been seen for a while, we remove it from the list of tracked objects as it is no longer of interest.
차량 물체가 위의 기준을 충족하면 카운터로 전달하기 전에 이동하는 방향을 확인하려고합니다. 그런 다음 이 정보를 사용하여 차량을 계수해야 하는지 여부를 결정한 다음 계수가 왼쪽 차선(위쪽 방향) 또는 오른쪽 차선(아래쪽 방향)에 적용되는지 여부를 결정할 수 있습니다. 만족하면 카운터를 업데이트하고 출력 프레임에 인쇄합니다.
차량이 한동안 보이지 않으면 더 이상 관심이 없으므로 추적된 개체 목록에서 제거합니다.


## Challenges and Improvements
The algorithm used works well in situations where traffic is free-flowing, within day-light hours. It also works relatively well in most weather conditions although background removal proves difficult in high winds as a moving camera means the background also changes quickly. However, accuracy drops when vehicles are either close together or have large shadows (forming one large object), dark vehicles do not always meet the detection criteria, and night scenes are difficult to resolve as headlight beams can create large areas that meet threshold criteria. Detection criteria are also relatively unique for each camera and so it may take time to refine these values to be confident in the output counts.
사용 된 알고리즘은 낮 시간 내에 트래픽이 자유롭게 흐르는 상황에서 잘 작동합니다. 또한 움직이는 카메라가 배경도 빠르게 변경된다는 것을 의미하기 때문에 강풍에서는 배경 제거가 어렵지만 대부분의 기상 조건에서 비교적 잘 작동합니다.
그러나 차량이 서로 가까이 있거나 큰 그림자가 있는 경우(하나의 큰 물체를 형성), 어두운 차량이 항상 감지 기준을 충족하는 것은 아니며, 헤드라이트 빔이 임계값 기준을 충족하는 넓은 영역을 생성할 수 있으므로 야간 장면을 해결하기 어렵습니다. 감지 기준도 각 카메라마다 상대적으로 고유하므로 출력 카운트를 확신하기 위해 이러한 값을 구체화하는 데 시간이 걸릴 수 있습니다.


Many of these issues could be resolved by investigating alternative detection methods that do not rely so heavily on detecting pixels above a threshold value. To that end, detecting vehicles using HAAR cascades would potentially resolve these issues or at least provide a more accurate and consistent method for counting vehicles in various conditions and without worrying too much about initial detection values. That said, this would create the need for good training data and potentially data for each camera and so would add be more resource heavy initially.
이러한 문제의 대부분은 임계값 이상의 픽셀 감지에 크게 의존하지 않는 대체 감지 방법을 조사하여 해결할 수 있습니다. 이를 위해 HAAR 캐스케이드를 사용하여 차량을 감지하면 잠재적으로 이러한 문제를 해결하거나 최소한 다양한 조건에서 차량을 계수하는 보다 정확하고 일관된 방법을 제공할 수 있습니다.
초기 검출 값에 대해 너무 걱정하지 않아도됩니다. 즉, 이것은 좋은 훈련 데이터와 잠재적으로 각 카메라에 대한 데이터에 대한 필요성을 창출하므로 처음에는 더 많은 리소스를 추가할 것입니다.

## Resources / Useful Reading
* [Counting Cars Open CV (Dan Masek)](https://stackoverflow.com/a/36274515)
* [Speed Tracking (Ian Dees)](https://github.com/iandees/speedtrack)
* [SDC Vehicle Lane Detection (Max Ritter)](https://github.com/maxritter/SDC-Vehicle-Lane-Detection)



https://github.com/seanlab1/traffic_counter

https://www.python.org/downloads/release/python-370/
python3.7 installl

pip install opencv-python  ==> v4.4.0.46
pip install opencv-contrib-python

black & white
https://www.videotoconvert.com/black-white/

충주 CCTV정보

https://its.chungju.go.kr/traffic/cctv.do#
