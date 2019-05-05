# couplet_generation

## Description
couplet generation system(对联生成系统)

## Dependencies
- Python3
- Tensorflow


## Dataset and Pretrained Model
You can download dataset and weights from:

- Baidu Yun: https://pan.baidu.com/s/1s8blKrTE7GT6OGZKae9fSw 提取码: jst4

or

- Google Drive: https://drive.google.com/open?id=1KNswtLQ7GsXdbnyoO6Pf2VfuYIg55HOT

, and put them under `./data/` and `./weights/`

## Pretrained Model

You can download pretrained from , and put them under `./weights/`
 
### Methods

#### Hidden Markov Model(HMM)

`python demo_HMM.py`

#### Maximize Entropy Markov Model(MEMM)

`python demo_MEMM.py`

#### Sequence to Sequence(LSTM)

`python demo_lstm.py`

#### Some results of the model:
---
- input:古今奇观属岩壑
- HMM output:天地美景如烟霞
- MEMM output:天地美景聊千山
- lstm output:中外名山胜故乡

---

- input:青山不墨千秋画
- HMM output:碧水无情万里春
- MEMM output:碧水长风万里春
- lstm output:绿水无弦万古琴
- review：通过这个对比,可以看出神经网络更具有诗意。神经网络的输出把绿水当做琴,和上
联的把青山当做画有个非常巧妙的照应。细细品味,让人置身于一个有声有色的山水
画当中,耐人寻味!
---
- input:两岸凉生菰叶雨
- HMM output:一轮明月桂花香
- MEMM output:一江春色千秋风
- lstm output:一江春涨藕花风
- review：通过对比可以看出,神经网络对词性的分析比其他两个模型更强。“凉生”是动词,
只有神经网络才正确的分析出来,且对应了“春涨”。“菰叶雨”中,也只有神经网
络给出了比较好的对应。且“春涨”与“藕花风”相互呼应,让人感受到春暖花开,
小荷才露尖尖角的小清新。
---
- input:春眠不觉晓
- HMM output:国梦难知春
- MEMM output:秋月方知春
- lstm output:花落自知秋
---
- input:无边落木萧萧下
- HMM output:万里飞花点点头
- MEMM output:不尽长江滚滚来
- lstm output:不尽流云处处游
- review：这里 MEMM 竟然给出了原诗的下联,而神经网络能给出一个合理的且非原诗的下联,
可见其泛化能力和强大的创作能力。
---
- input:两只黄鹂鸣翠柳
- HMM output:一方紫燕舞红梅
- MEMM output:一群紫燕舞红梅
- lstm output:一弯新月钓清溪
- review：这里三个模型都给出合理的下联,但是我觉得神经网络的下联更加的富有诗意。我们
都知道,新月相比满月在于它是像弯钩一样,它的倒影在小溪中,如同想要从溪水中
钓鱼一般,想象力非常丰富。而其他两个模型的输出就少了这般诗意。
---
- input:万紫千红春无限
- HMM output:千红万紫燕有余
- MEMM output:五湖四海风有余
- lstm output:五风十雨岁有余
- review：这里看出,神经网络的输出更具有祝福气息,更符合新春的氛围。五风十雨在百度百
科的解释为:五天刮一次风,十天下一场雨,形容风调雨顺。就是祝福大家风调雨
顺,收获满满,岁岁有余。可以说神经网络是对词语的意思理解得非常透彻了。
---
- input:深秋帘幕千家雨
- HMM output:静夜月光万里风
- MEMM output:明月镜头万里风
- lstm output:仲夏麦田万亩稻
- review：可以看出,HMM 和 MEMM 的输出可谓前后不搭,而神经网络的输出,“麦田”与
“稻”遥相呼应,“千家雨”与“万亩稻”也对应得非常好。
输入:月透柳帘窥案卷
HMM 输出:花开花院落花开
MEMM 输出:花飞花果三千秋
seq2seq 输出:风吹竹管动窗纱
---

## Train via yourself
By running `python train_HMM.py`,`python train_MEMM.py` and `python train_lstm.py`, you can train a model by yourself.


