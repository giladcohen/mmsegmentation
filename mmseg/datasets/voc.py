# Copyright (c) OpenMMLab. All rights reserved.
import os.path as osp
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset

from research.utils import generate_farthest_vecs


@DATASETS.register_module()
class PascalVOCDataset(CustomDataset):
    """Pascal VOC dataset.

    Args:
        split (str): Split txt file for Pascal VOC.
    """
    EMB_DIM = 200

    CLASSES = ('background', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
               'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa',
               'train', 'tvmonitor')

    PALETTE = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0], [0, 0, 128],
               [128, 0, 128], [0, 128, 128], [128, 128, 128], [64, 0, 0],
               [192, 0, 0], [64, 128, 0], [192, 128, 0], [64, 0, 128],
               [192, 0, 128], [64, 128, 128], [192, 128, 128], [0, 64, 0],
               [128, 64, 0], [0, 192, 0], [128, 192, 0], [0, 64, 128]]

    def __init__(self, split, **kwargs):
        self.emb_selection = kwargs.pop('emb_selection', None)
        super(PascalVOCDataset, self).__init__(
            img_suffix='.jpg', seg_map_suffix='.png', split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
        self.idx_to_class_emb_vec = self.set_emb_vecs(self.emb_selection)
        print('cool')

    @staticmethod
    def parse_vec(s: str):
        return np.asarray(list(map(float, s.split())))

    def set_glove(self):
        gloves = np.zeros((len(self.CLASSES), self.EMB_DIM), dtype=np.float32)

        # airplane (instead of aeroplane)
        gloves[1] = self.parse_vec('-0.42252 -0.72499 0.3823 -0.28675 -0.070732 1.082 0.61925 -0.51744 -0.24192 0.36525 -0.10519 0.68813 -0.82839 0.0121 -0.30335 0.057322 0.077832 0.11161 0.46033 -0.21916 -0.049768 -0.24293 0.12415 -0.40696 0.32383 1.0217 0.62564 -0.75066 -0.41027 -0.0758 -0.1808 -0.027986 0.21466 -1.1386 0.20759 0.67844 -0.60843 0.28039 1.0015 0.014468 0.2675 -0.10874 -0.23052 -0.83247 0.2413 -0.11418 -0.31517 -0.28662 0.067465 0.17122 0.16358 -0.38727 -0.33752 0.15207 0.071406 -0.23285 -0.39292 0.79661 -0.01181 -0.61611 0.42596 -0.024823 0.51229 -0.1942 -0.31514 -0.9923 0.26809 -0.16498 0.20328 -0.21459 -0.70433 -0.0017985 -0.65342 -0.85474 0.161 -0.71959 -0.50075 -0.18926 0.31129 0.90581 0.58413 0.87044 -0.056666 -0.26441 0.29036 0.07847 0.026343 0.3536 -1.1024 0.4081 0.26188 -0.20925 -0.728 -0.04421 -0.21305 -0.2336 -0.33843 -0.27006 -0.81843 -0.19834 0.58124 0.039614 -0.90533 0.39462 -0.35865 -0.47045 0.22981 -0.044953 0.28625 -0.14308 -0.31557 -0.015199 -0.28968 -0.28257 -0.72873 -0.13707 -0.0014256 -0.44722 -0.14099 -0.062103 0.53414 -0.18197 -0.13406 -0.41105 -0.39153 0.73264 0.031486 0.3796 0.40439 0.37544 0.49086 0.38665 0.095826 0.2573 -0.47709 -0.5425 0.19142 0.66534 -0.26036 0.044465 -0.1965 0.21443 0.090587 0.48187 0.063059 0.10099 0.23694 -0.16066 -0.39295 -0.62392 1.2988 -0.2949 -1.8037 0.32934 -0.11134 0.0236 0.29623 -0.39351 0.058452 -0.37467 -0.029277 0.073365 0.3801 0.67572 0.10034 -0.27386 -0.58898 0.18683 0.029444 0.20757 0.01653 -0.4761 0.15124 -0.24604 0.064738 0.22999 -0.80299 0.20186 -0.012943 0.80957 0.25185 -0.28367 -0.0093086 0.2747 -0.91049 0.24138 0.31127 -0.084327 0.15578 -0.23792 0.74639 -0.24335 -0.084517 -0.072658 0.027183 0.083656 0.10962 0.025677 0.26856 0.049582')
        # bicycle
        gloves[2] = self.parse_vec('-0.20953 0.71027 0.20456 0.030102 -0.15586 -0.0017965 0.64207 -0.7232 0.6517 -0.50303 0.46756 0.38291 -1.0521 -0.35768 -0.093636 -0.026939 -0.15788 -0.079467 0.56652 0.34809 0.92798 -0.26454 0.89587 -0.18168 -0.2479 1.0151 0.1562 0.29677 -0.24759 -0.32084 0.18955 -0.15548 0.54911 -0.67806 0.47885 0.027665 -0.26085 0.22484 0.48099 0.058068 0.55766 0.084012 -0.051574 0.40012 0.31149 -0.34196 0.091012 0.32463 0.48642 0.49414 0.16418 -0.62328 -0.41107 -0.56187 0.57129 0.045219 0.095414 0.12472 0.19763 0.10691 -0.12217 0.10911 0.094954 -0.20152 -0.17483 -0.42543 0.16613 0.58936 -0.095105 -0.44676 -0.36252 0.10529 -0.36694 0.103 0.21666 0.17183 -0.45817 0.25464 -0.066484 -0.39853 0.59684 0.24154 -0.46958 -0.42001 0.19729 0.45703 0.9437 0.19471 -0.28348 0.26896 0.17337 -0.28803 -0.21938 -0.04655 -0.45331 0.21835 -0.11856 0.13973 0.085915 0.59576 0.83806 0.44316 0.061283 -0.023769 -0.54969 -1.5631 -0.30484 -0.5664 0.17964 0.29822 0.67106 0.29276 0.32477 0.35384 0.12617 0.65192 -0.64218 -0.38125 -0.52421 0.66719 0.61863 -0.29273 -0.2346 0.22393 -0.29474 -0.44579 -0.025123 1.3639 -0.09371 0.4203 1.0943 0.54408 0.28939 -0.42816 -0.44594 0.49912 -0.24159 0.27606 -0.14985 0.13104 -0.39032 0.26478 0.03135 0.18696 -0.39013 -0.0049679 0.50424 -0.36814 0.17211 0.68211 0.6758 -0.56006 -1.8271 0.35589 0.007275 -0.056845 -0.12371 0.62302 0.25987 -0.27712 0.66312 0.49514 0.3868 0.20792 -0.24442 -0.03075 -0.042747 -0.099471 0.076467 0.0563 0.81152 -0.2869 -0.53929 -0.16035 0.71853 0.59261 0.050601 -0.62398 0.25599 0.37793 0.56556 0.24721 0.32267 -0.34488 0.035321 -0.45666 0.083282 -0.74305 0.25377 0.26414 0.53079 -0.27572 0.12793 0.22571 0.028646 -0.22586 0.5593 0.15792 0.002043 0.15446')
        # bird
        gloves[3] = self.parse_vec('0.050286 -0.40342 -0.085693 -0.11261 -0.40626 0.27764 0.28587 -0.036815 0.29082 0.53717 -0.096179 0.20294 -0.52494 -0.42556 -0.020042 0.59147 0.60556 -0.096592 0.078125 -1.009 -0.48508 0.26272 -0.36493 -0.72437 0.044094 0.46839 0.22695 0.080163 -0.18623 0.49568 -0.067437 0.29948 -0.36965 -0.73587 -0.033697 0.35647 -0.13801 0.42026 -0.064175 -0.35642 -0.40864 0.081728 0.1202 -0.45304 0.35192 -0.16238 -0.40587 0.28837 0.72754 0.5276 -0.12201 -0.18372 0.36878 0.46526 0.32681 -0.56752 -0.50191 0.60814 0.57881 0.0227 0.23608 0.035366 0.16645 -0.028746 -0.13858 -0.42193 0.42848 -0.011398 0.32289 0.204 -0.34057 0.30971 -0.5685 -0.85169 -0.12805 -0.3842 -0.11821 0.050055 0.50502 0.58767 1.0039 0.3996 -0.027687 0.17466 -0.22844 0.12718 -0.51194 -0.45218 -0.20525 0.055035 0.27 -1.0207 -1.1003 -0.51314 -0.35455 -0.13669 -0.17903 0.10799 -0.24093 0.66859 -0.13704 0.50379 -0.065461 0.15555 -0.51893 0.62364 -0.52682 0.16933 -0.44093 -0.090353 -0.84958 0.42558 -0.31874 -0.38313 0.39895 -0.067433 1.0144 -0.17431 -0.063368 -0.60363 0.20053 0.13679 -0.024741 0.47469 -0.77892 -0.28663 -0.27192 -0.67562 0.28207 0.1935 0.063162 0.73112 0.072682 0.51456 -0.55077 -0.25402 -0.077662 0.035238 -0.32021 -0.33759 -0.24357 0.035842 0.81423 -0.3508 0.18006 -0.049245 0.12888 -0.16803 -0.3665 0.63389 -0.13232 -0.54769 -3.4213 -0.38828 -0.24938 -0.41294 -0.2727 -0.3304 0.23315 -0.52551 0.21471 -0.38583 -0.30177 0.30061 -0.33541 -0.60107 0.23551 -0.80369 -0.13737 -0.1429 0.16166 0.32293 -0.12294 0.16138 -0.093296 0.14234 0.27728 0.036312 -0.19796 0.1936 -0.46891 0.82351 -0.53899 -0.24703 0.049887 0.54725 0.009746 0.57974 -0.0091502 -0.34196 0.026213 0.19177 0.5079 0.16918 0.6699 0.4473 -0.61384 -0.015805 -0.42108 -0.087537')
        # boat
        gloves[4] = self.parse_vec('-0.39539 -0.25468 0.043564 0.36511 0.18522 0.329 0.19064 -0.11648 0.31226 -0.040298 -0.0062365 0.30342 -0.42173 0.77493 -0.03998 -0.067118 0.13732 0.95702 0.40353 -0.33322 -0.59533 -0.12267 0.12258 -0.042508 -0.14386 1.1716 0.39072 -0.047285 -0.0033427 -0.81392 0.72796 0.052686 -0.049161 -0.71438 -0.086344 0.33522 0.14088 0.70827 0.2561 0.16326 -0.006642 0.090248 0.16412 0.17618 0.47049 0.018178 0.77729 -0.39745 0.71365 0.64572 0.26825 -0.00055794 -0.76011 -0.37583 -0.20395 -0.083587 -0.49212 0.35199 -0.091585 -0.42059 0.166 1.0091 0.11889 -1.0233 -0.25455 -0.0037728 -0.31496 -0.0079189 -0.00569 -0.94841 -0.24254 0.00080959 0.65628 -0.54486 0.6096 -0.38037 -0.78455 0.12337 0.72398 -0.31379 1.1729 -0.18303 0.10475 -0.04287 -0.27979 -0.10889 0.3874 0.11326 -0.15383 0.32006 -0.11064 -0.1193 -0.33176 -0.31274 -0.11912 0.16069 -0.037982 0.23802 -0.91678 0.30449 0.60797 0.073835 -0.26335 -0.029634 -1.05 0.20826 -0.21924 0.13652 0.40489 0.25212 -0.22705 0.26812 -0.1994 -0.53777 -0.4988 -0.47727 -0.66004 -0.83413 0.047445 0.15756 0.23355 0.21463 -0.056451 0.080833 -0.044144 -0.046193 0.020127 0.61713 0.23021 0.46089 0.45184 -0.053696 -0.29686 0.065724 -0.2795 0.38674 0.10408 0.34197 -0.55379 -0.67967 -0.47101 0.35917 0.1974 -0.043696 -0.052605 -0.73159 0.16067 -0.3786 0.2434 0.61161 0.48916 -0.57555 -3.0066 -0.3901 0.38596 -0.048683 0.39269 0.6831 0.64456 0.87903 0.21022 0.17747 0.017671 0.60079 -0.41003 -0.26996 -0.0044936 0.14928 -0.40555 -0.16593 -0.85092 0.027109 -0.40114 -0.038453 0.17137 -0.17077 -0.17581 0.11836 0.223 0.59717 0.36317 -0.035388 0.30407 0.53003 -0.090254 -0.50943 0.28771 -0.1125 -0.35207 -0.07374 0.57425 -0.60225 -0.19009 0.43454 0.64101 -0.081903 0.529 0.15899 0.021136 -0.016624')
        # bottle
        gloves[5] = self.parse_vec('-0.79897 0.12251 0.15633 -0.023137 0.20395 -0.40863 0.11329 -0.26234 -0.04337 -0.28863 0.32162 0.80217 -0.69404 0.072699 0.032425 0.081859 0.49708 0.44474 -0.20787 0.10049 -0.36369 -0.020898 -0.0027382 -0.61522 0.38828 1.4885 -0.031765 0.27525 0.4149 0.13678 0.032849 0.094527 -1.2946 -0.14829 -0.75905 0.21244 0.11954 -0.25734 0.21472 0.11741 0.23785 0.23741 -0.32102 -0.16134 0.21676 0.05692 0.3519 0.57165 -0.13035 0.25762 -0.13437 0.048592 0.069208 -0.12793 0.08571 -0.17723 0.75061 0.074342 -0.63924 -0.046564 0.18867 -0.22023 -0.12546 -0.53414 0.21347 -1.2106 -0.14119 -0.62831 0.80332 -0.020454 0.21436 -0.5496 0.38633 0.36767 0.26217 -0.33457 0.10184 0.025629 0.01278 -0.0032671 1.1778 0.25938 -0.15306 -0.96678 0.5922 0.69536 -0.28397 0.082051 -0.4951 1.5883 -0.47416 0.017795 -0.041617 -0.5739 0.10164 -0.25656 -0.37935 -0.0095207 -0.29664 0.33145 -0.20419 -0.18354 -0.066054 0.56563 -0.8608 -0.54741 0.14342 -0.7112 -0.76279 -0.50002 -0.69331 0.75902 0.05013 -0.75578 0.058621 -0.36132 -0.57238 -0.18413 -0.10716 0.1963 -0.28295 0.13177 0.37334 -0.49856 0.085692 0.14263 0.040408 0.46739 0.47784 -0.35338 0.032038 -0.31784 -0.53549 -0.49545 -0.24752 0.082921 -0.22467 -0.093533 0.20728 0.49855 -0.056853 0.15364 -0.11297 -0.5746 0.33484 -0.55111 0.74624 0.21023 -0.20434 0.19723 0.6313 -0.2206 -3.2973 0.40132 -0.045925 0.031596 -0.19902 0.52396 0.18297 0.2443 0.30136 -0.26096 0.4531 -0.36779 -0.019241 0.17294 -0.69498 0.31856 0.10471 0.47494 0.11335 0.68598 -0.37452 0.053953 -0.72787 -0.50056 -0.33375 0.64967 0.29411 0.48564 0.034691 -0.04236 -0.02612 -0.10335 -0.27702 0.018744 -0.021129 -0.73097 -0.15203 -0.11875 -0.15249 -0.15179 -0.53379 0.75922 0.92714 -0.14741 0.26636 -0.23923 0.84491 -0.7012')
        # bus
        gloves[6] = self.parse_vec('0.36878 -0.040716 0.14877 -0.16091 0.25884 0.42093 -0.08497 -0.20741 -0.24405 0.16025 0.18248 0.27653 0.17274 -0.15511 0.1832 -0.59696 0.35511 0.21179 -0.88778 -0.14127 0.27427 -0.22426 -0.49829 -0.2489 -0.64608 -0.51976 0.029963 -0.39474 -0.3698 -0.45758 -0.26379 0.0055427 0.072394 -0.4574 0.13783 0.41553 -0.71718 0.36648 0.80797 0.11551 -0.44923 0.33793 -0.38741 -0.55758 0.064246 0.040185 -0.13671 0.15378 0.41823 0.33495 0.265 -0.18855 -0.20561 -0.56125 -0.49499 -0.29046 -0.38711 -0.040435 -0.60069 -0.37021 0.40149 -0.15775 0.64168 -0.027062 0.43667 -0.3754 0.2332 0.4121 -0.3158 -0.1494 -0.23384 -0.013539 0.25869 -0.56107 -0.29731 0.56592 -0.13422 0.012458 0.19112 0.35151 0.3017 -0.63447 -0.020045 -0.027795 -0.0084391 0.27444 -0.13512 -0.70592 0.64869 -0.32654 0.13714 -0.43252 -0.1321 0.32763 0.043845 0.2212 0.18353 -0.15674 -0.50952 0.15471 0.60796 -0.63089 0.29242 -0.37111 -0.31205 -0.91168 0.4415 -0.25655 0.36425 0.097246 -0.55528 0.29396 0.45414 -0.10683 -0.17456 -0.3311 -0.10974 -0.32565 0.10095 0.74103 0.3077 -0.60567 -0.34343 -0.08782 -0.36266 0.63673 0.17799 0.61259 -0.18688 0.80418 0.42218 -0.20539 0.14961 -0.30303 -0.79753 0.10696 -0.35002 0.23048 0.15042 0.061245 -0.59652 0.0026576 0.05751 0.034295 0.024454 0.097094 -0.0058212 -0.79352 -0.43982 -0.45078 0.33703 -0.081068 -4.0471 0.21823 0.20914 -0.66168 -0.010194 0.86391 0.31894 0.0099252 0.69654 0.4219 0.68502 0.26832 -0.31542 -0.60462 -0.89089 -0.27853 -0.28233 -0.22141 -0.31363 -0.045722 -0.78919 -0.42835 0.90955 -0.49916 0.20697 0.036049 -0.38361 0.69864 0.58477 -0.12021 -0.14528 0.61904 -0.39795 0.042507 -0.04765 0.37876 0.54698 0.26489 0.6039 -0.48082 0.017844 0.4663 -0.35059 -0.098496 -0.5092 0.43729 -0.3703 0.73458')
        # car
        gloves[7] = self.parse_vec('-0.023756 -0.6095 -0.64204 0.21877 0.46728 0.18328 -0.017327 -0.1671 0.15519 -0.19869 0.58117 0.40394 -0.39322 -0.14633 -0.14179 0.015474 0.11165 -0.10333 -0.20328 -0.071406 0.12644 -0.26139 -0.36218 -0.67246 -0.34604 0.59822 -0.17553 -0.031497 0.11128 -0.3225 0.061777 0.38997 -0.33846 -0.1767 -0.082802 0.41319 -0.47078 0.48865 0.74484 0.24344 0.43444 0.34383 -0.63643 0.41448 -0.38013 -0.16224 0.41776 -0.045915 0.76219 0.055854 0.80065 0.22815 -0.95708 -0.064152 -0.25136 0.030722 -0.56599 0.13781 0.093393 -0.83462 0.32205 -0.065024 0.86411 -0.054507 0.19187 -0.39785 0.16377 0.57524 -0.37361 -0.72036 -0.48547 0.18768 -0.2428 -0.0031741 -0.43129 0.21333 -0.36452 0.15536 -0.18761 0.43804 0.66989 0.1977 -0.48026 0.17955 -0.26623 0.3866 0.37762 0.33181 -0.29401 0.089559 -0.1417 0.090185 0.23631 0.05726 0.49807 0.5556 0.0085019 -0.19751 -0.99868 -0.12837 0.72538 -0.21058 -0.17776 0.54406 -0.51257 -0.30398 0.5172 -0.4982 0.72498 -0.13728 -0.15657 0.48735 -0.12313 -0.44957 0.10629 0.13345 -0.71389 -0.41793 -0.77205 0.70404 0.35033 -0.33719 -0.23397 -0.18326 -0.36967 0.76203 0.23946 0.85417 0.069386 -0.19864 0.38917 -0.12225 -0.34538 0.062926 -0.31898 0.17836 -0.4046 0.38409 -0.20409 0.35095 -0.42669 -0.06645 0.2125 0.14951 -0.23864 0.1338 0.11083 0.21279 -0.0037618 -0.13022 0.21465 -0.51508 -4.7217 0.15789 0.26162 -0.15878 0.012484 -0.13879 0.40189 -0.49206 0.35261 0.62121 0.37681 0.54427 0.06366 -0.3226 -0.47194 -0.6409 -0.16708 -0.067091 0.21019 0.52271 -0.51378 -0.45009 0.77929 -0.033527 0.34275 0.15728 0.22613 1.0059 0.091323 0.025024 0.1937 0.17346 0.35938 -0.59598 0.52244 -0.32664 0.23388 0.29734 -0.1782 -0.58709 0.58139 -0.39022 -0.17797 0.02756 -0.2737 0.00032772 0.3212 0.31734')
        # cat
        gloves[8] = self.parse_vec('0.14557 -0.47214 0.045594 -0.11133 -0.44561 0.016502 0.46724 -0.18545 0.41239 -0.67263 -0.48698 0.72586 -0.22125 -0.20023 0.1779 0.67062 0.41636 0.065783 0.48212 -0.035627 -0.47048 0.077485 -0.28296 -0.49671 0.337 0.71805 0.22005 0.12718 0.067862 0.40265 -0.01821 0.78379 -0.52571 -0.39359 -0.56827 -0.15662 -0.084099 -0.20918 -0.066157 0.25114 -0.40015 0.1593 0.17887 -0.3211 0.09951 0.52923 0.48289 0.14505 0.44368 0.17365 0.3635 -0.51496 -0.12889 -0.19713 0.18096 -0.011301 0.84409 0.98606 0.83535 0.3541 -0.23395 0.3551 0.41899 -0.054763 0.22902 -0.19593 -0.57777 0.29728 0.33972 -0.31119 -0.32498 -0.42557 -0.70302 -0.72515 -0.29349 0.49964 -0.32889 0.24359 0.13243 0.31164 1.2156 0.31241 -0.23794 0.38422 -0.321 -0.28756 -0.20047 0.34454 -0.64929 0.28021 0.060203 0.053618 -0.13341 0.2451 0.18639 -0.0016346 -0.066883 0.077845 -0.085217 0.75257 0.76264 -0.053318 0.071056 0.30552 -0.43411 -0.19361 -0.10493 -0.53732 -0.239 -0.47298 -0.029825 -0.20206 -0.48945 -0.13616 0.49622 0.20743 -0.077396 -0.34304 0.0062387 -0.0065902 -0.24729 -0.013859 -0.079919 0.43452 0.23415 0.17995 0.13236 -0.22717 -0.55278 0.042005 0.21937 0.42042 0.43639 -0.58305 -0.118 0.15379 -0.29596 -0.46251 0.52593 0.10471 -0.19973 -0.028228 0.49974 -0.58053 -0.51416 0.21325 -0.38394 -0.00059821 0.16525 -0.055993 -0.4008 -0.05483 -3.8842 -0.022136 -0.46989 0.23502 0.081298 0.83091 0.47251 0.074057 0.15737 0.065809 -0.26756 0.1947 -0.63597 -0.59914 -0.21369 0.011718 -0.25464 -0.19629 0.18017 0.59031 0.0062176 0.51122 0.36601 -0.27381 -0.11342 0.21195 0.43099 -0.43837 0.12842 0.39312 -0.19492 0.056414 0.54343 0.13678 -0.71087 0.38758 -0.0078956 -0.32383 0.064193 -0.22329 0.071366 -0.30966 -0.46142 0.29545 -0.49186 0.24053 -0.46081 -0.077296')
        # chair
        gloves[9] = self.parse_vec('0.083778 -0.31358 0.44036 -0.19852 0.43794 0.51642 0.53045 0.38768 -0.25435 -0.13987 -0.087003 0.52748 -1.0245 0.26502 0.39768 -0.080842 -0.22176 0.25287 -0.22036 0.19245 0.31503 0.24298 -0.31244 -0.5538 -0.065636 1.1332 0.59765 -0.044034 -0.78153 -0.86698 0.28703 -0.76905 -0.084277 -0.22998 -0.15668 -0.3007 0.3213 0.056273 0.28742 0.2602 0.84825 -0.0071684 0.37892 -0.012884 0.00038517 0.17809 0.63603 0.89252 0.3586 0.20689 0.46894 -0.53883 0.0010013 -0.040398 0.0050846 -0.088845 0.40522 -0.00066163 0.40549 0.078797 0.22208 0.28788 0.7882 -0.70755 -0.39356 -0.29528 0.40909 -0.36923 0.72393 -0.17285 0.097639 -0.028392 -0.028554 -0.18386 -0.21958 0.41438 0.12902 0.29108 -0.49385 0.30497 0.020471 0.10858 -0.44766 -0.072593 0.50049 -0.34468 0.45321 0.1845 -0.35328 0.43199 -0.11018 0.26425 -0.63166 0.11634 0.67827 -0.57504 0.16556 -0.88157 -0.94127 0.35106 0.15176 -0.13839 0.12987 -0.33697 -1.1608 -0.14715 -0.054598 -0.32148 0.070592 -0.30956 -0.07437 0.76935 0.19682 -0.59907 -0.10843 0.39593 0.11362 -0.85316 0.10575 0.25386 -0.0021121 0.47077 -0.11135 0.35682 -0.13714 0.21096 0.058276 0.55903 0.25444 0.32109 0.35921 0.66993 -0.59417 -0.043362 -0.12672 -0.66172 -0.0062734 0.6619 0.13831 0.63765 -0.42123 -0.26323 0.13225 -0.62235 0.42746 -0.32953 0.17725 -0.2127 -0.13381 0.39902 -0.24999 -0.28896 -2.8864 0.3831 0.091285 0.35551 0.3535 0.061948 0.35884 -0.020577 0.19219 -0.018047 0.88794 0.11279 0.25829 0.14008 -0.00049045 0.33372 0.10877 -0.20534 0.49567 0.18442 -0.51278 0.39767 0.95853 -0.38023 -0.01555 0.52021 -0.40211 0.38038 0.25662 0.11418 0.833 -0.039078 0.19066 0.15591 -0.45687 -0.12533 0.96457 -0.77102 0.42057 -0.37074 0.20668 0.32806 -0.12334 -0.38058 0.66554 0.10284 -0.38228 -0.26866')
        # cow
        gloves[10] = self.parse_vec('-0.50022 -0.36807 0.67852 0.73902 -0.265 0.2138 0.80012 -0.32307 -0.022903 -0.095265 -0.049275 0.85775 -0.1414 -0.23757 0.53613 0.76321 0.63271 -0.98486 0.21919 -0.45295 0.63721 0.11644 -0.6411 -0.14992 0.22396 1.0825 -0.09032 0.063134 0.09663 0.39048 0.12483 0.52111 -0.30639 -0.11429 -0.36173 0.20997 -0.32267 0.3406 0.095895 -0.046656 0.34377 -0.12895 -0.6377 0.35499 0.095412 0.26032 0.11898 0.32955 1.1196 0.10973 0.15534 -0.12486 -0.35955 -0.013375 0.41262 -0.37091 0.62772 0.44115 0.11786 0.5494 -0.79519 0.58553 0.09613 0.076929 -0.19485 -0.094721 -0.40216 0.47339 0.031281 0.56596 -0.096632 -0.28741 -0.058642 -0.60075 -0.258 0.11909 -0.31724 0.21365 -0.036304 0.40186 0.28296 0.60792 -0.64312 0.25329 -0.82223 0.64957 -0.15475 -0.057517 -0.048461 0.31191 -0.46918 -0.29295 0.2265 0.15877 0.21139 -0.077235 0.37437 -0.14858 0.30027 0.48047 -0.098092 0.46117 -0.21483 0.13998 -0.83095 -0.45552 -0.11837 -0.11443 -0.31663 -0.79722 0.058454 -0.23475 0.066028 0.22309 0.14601 -0.044701 -0.33712 0.63045 -0.16638 -0.67182 -0.2189 -0.14132 -0.043728 0.54265 -0.37985 0.059618 0.075789 -0.55127 -0.27159 0.11659 0.3785 0.16998 0.66348 0.20145 -0.097833 0.18527 -0.097937 0.64232 -0.40563 -0.21788 -0.35083 0.52864 -0.20921 -0.98088 0.066697 0.42067 0.13533 0.10734 -0.22574 -0.052797 0.041153 0.14589 -2.7129 -0.13888 0.10586 -0.37203 -0.043385 0.59728 0.34913 0.2266 0.094155 -0.23491 0.20874 0.063022 -0.13774 -0.61335 -0.55479 -0.032523 -0.35708 2.6989e-05 0.29623 0.44281 -0.29544 0.40348 0.030594 -0.48329 -0.44488 0.29776 0.19371 -0.068755 -0.53631 0.31017 -0.086424 0.11114 -0.055969 0.33717 0.077037 -0.062266 -0.19782 0.3087 -0.011787 -0.092054 0.49202 0.9067 -0.3875 -0.38298 -0.51466 0.27193 -0.46579 0.39654')
        # table (instead of diningtable)
        gloves[11] = self.parse_vec('-0.134 -0.33646 0.54234 -0.38614 0.35032 -0.042428 0.65948 0.50268 -0.23358 0.065875 -0.2383 0.3261 -0.88971 0.1316 0.1286 0.54411 -0.060063 -0.58494 -0.87027 0.068012 0.23148 0.060188 -0.34582 -0.5468 0.10941 0.51938 0.082787 0.22915 -0.0094834 0.040299 0.24899 -0.306 -0.22724 -0.58301 0.20897 -0.29863 0.61531 0.20226 0.88812 0.25077 0.37314 -0.081076 0.21412 0.23626 0.20637 0.13475 0.38395 0.23572 0.19801 0.34831 -0.29573 0.057377 0.22969 0.20866 0.67706 -0.3422 0.19446 -0.048101 0.062835 -0.35476 0.36633 0.26445 0.38393 -0.2259 -0.35441 -0.17699 0.49916 -0.39928 1.2351 0.087057 -0.12733 -0.17771 -0.33468 0.35263 -0.012405 0.030928 0.52244 0.058012 0.042316 0.65819 0.056759 -0.4262 0.022662 -0.933 0.60916 -0.12176 0.42021 -0.393 -0.23767 0.074235 -0.073421 0.88081 -0.72143 -0.38029 0.50629 0.0015509 0.10175 -0.53257 -0.56345 0.93009 0.02815 -0.13692 -0.15743 0.22503 -0.64667 -0.28772 -0.68087 -0.41039 0.070034 0.022488 -0.42095 -0.02085 0.0089226 -0.49268 0.20415 0.20063 0.47755 -0.47341 -0.070567 0.35511 -0.19021 0.55616 0.071037 0.48354 0.053282 0.194 0.64685 0.70101 -0.051358 -0.15977 0.54975 0.0050765 -0.088246 -0.20462 -0.68097 -0.36608 -0.45045 0.098466 0.039217 0.79404 -0.26734 -0.16116 -0.20512 -0.80283 0.52077 -0.27359 0.61654 -0.25623 -0.29343 0.20662 -0.60995 -0.48954 -3.5513 0.20977 0.37195 0.41746 0.24383 -0.25487 0.17495 0.085444 0.23693 -0.12911 0.040175 -0.15206 0.15921 0.2538 -0.092471 0.21385 0.81152 0.22078 0.36054 0.2941 -0.45904 0.12069 0.71867 -0.17193 0.25481 0.63885 -0.34664 0.58897 -0.23721 -0.15426 0.35082 -0.58878 -0.0075455 -0.20697 -0.38027 -0.53076 0.060267 -0.59977 0.16978 -0.18702 0.27114 -0.44326 0.171 0.067128 0.218 -0.10632 0.33975 -0.32446')
        # dog
        gloves[12] = self.parse_vec('-0.49586 -0.59369 -0.107 0.05593 -0.24633 -0.14021 0.63707 0.024992 0.25119 -0.55602 -0.37298 0.60131 -0.35971 -0.096752 0.18511 0.58992 0.47578 -0.16833 0.67079 -0.29472 0.069403 0.05334 -0.36154 -0.12883 0.27814 0.87467 0.12119 0.78215 -0.50617 0.28794 0.14213 0.83281 -0.27079 -0.28813 -0.67607 0.17991 -0.11046 -0.063062 -0.56297 0.36639 0.11009 0.2965 -0.12457 -0.11112 -0.24293 0.53344 0.75589 0.078154 0.91641 0.20878 0.01236 -0.71199 0.19085 -0.5199 -0.14181 0.078136 0.44157 1.0958 0.59009 0.35117 0.021684 0.1073 0.19942 -0.26355 0.084024 -0.32073 -0.24306 0.44821 0.14432 -0.063988 -0.15013 -0.33644 -0.67873 -0.64554 0.10706 0.64709 -0.20094 0.064682 0.035356 0.029288 0.99793 0.34343 -0.019469 0.70635 -0.54329 -0.057843 0.12624 -0.18132 0.099001 0.4478 -0.2641 -0.37506 -0.11238 -0.011805 0.33187 0.45295 0.1682 0.18379 0.29457 0.98963 0.5394 -0.0025833 -0.10989 0.30163 0.34495 -0.2275 -0.21093 -0.79685 0.29833 -0.64644 -0.18653 0.31771 0.061874 -0.44503 0.34052 0.5552 0.017743 -0.33609 0.18478 0.392 -0.44685 -0.2591 -0.4929 0.61712 -0.24546 0.15348 0.19796 0.041105 0.030167 0.13735 0.29154 0.079533 0.53594 -0.61848 0.082946 -0.43806 -0.16041 -0.44336 0.065162 0.29823 -0.13321 0.55445 0.29978 -0.63209 -0.45078 0.1534 -0.31124 0.258 0.062033 0.047879 0.37758 -0.007643 -4.328 0.65362 -0.45488 -0.4565 0.23566 1.0171 0.53344 -0.025861 0.067191 0.60342 -0.56511 0.57175 -0.47311 -0.43066 -0.13385 0.011506 -0.32674 -0.47726 0.010775 0.49053 -0.11302 0.23358 0.098286 -0.55746 0.096976 0.036503 0.41838 -0.22967 0.12346 0.23573 -0.17653 0.03863 0.62339 -0.083598 -0.62161 0.11059 0.11316 -0.26833 0.023406 -0.018887 -0.63446 -0.16513 -0.16886 0.087242 -0.10353 0.06788 -0.20546 0.17962')
        # horse
        gloves[13] = self.parse_vec('-0.8107 -0.2135 0.57229 0.38901 -0.53731 0.076275 0.80555 -0.64481 0.58122 -0.003714 0.15482 0.5188 -0.73224 -0.17708 0.37883 1.0903 0.39686 -0.38992 0.45664 -0.31646 0.49369 -0.16371 -0.45948 -0.21822 0.34105 0.96526 0.25932 0.12078 0.012586 0.084278 0.50996 0.27742 -0.15154 -0.13721 -0.098856 0.12999 -0.41539 0.21986 -0.27817 -0.1278 0.1805 -0.71333 0.3577 0.42558 0.25589 0.443 0.36289 0.17151 1.0117 0.74856 0.26782 -0.029225 -0.36808 -0.13197 0.51501 0.13333 0.0058557 0.80578 -0.0721 0.70669 -0.50893 1.2565 0.20282 -0.13758 -0.5108 -0.34195 -0.24551 0.53538 0.2398 -0.30907 -0.20728 -0.82592 -0.34368 0.017876 0.092939 0.049257 -0.43085 -0.13684 0.019521 -0.20954 0.58053 -0.18977 -0.28645 0.44486 -0.5442 0.708 0.46365 0.086484 -0.042811 0.04067 -0.26089 -0.4174 -0.35112 -0.45257 0.27432 0.42729 0.4371 0.31975 0.017235 0.42254 -0.053444 -0.16006 -0.31785 0.33874 -0.23682 -0.34646 -0.30786 -0.55616 -0.045204 0.012021 -0.63051 0.3996 -0.29002 0.0079054 0.047329 0.5004 0.060087 -0.2037 0.12378 0.24339 -0.38377 -0.50928 -0.1049 0.14504 -0.39883 -0.24158 -0.33095 0.20819 0.81785 -0.34484 0.25812 0.017235 0.25583 -0.096405 0.16331 0.12816 -0.1257 0.11052 -0.19591 0.26462 -0.093251 0.74641 0.37195 -0.19395 -0.26052 -0.36437 0.46078 0.22374 -0.15367 0.3202 0.19659 -0.18048 -3.2003 0.24416 -0.36079 -0.022701 -0.10411 0.57065 0.20385 0.020388 0.78644 0.55647 -0.1408 -0.11196 -0.50173 -0.38527 -0.2307 0.062547 -0.54328 -0.56776 0.38209 0.10156 -0.16395 0.35198 0.55722 -0.34555 0.017989 -0.040839 0.28383 -0.049434 0.11944 0.086508 0.4774 0.073957 -0.23412 0.29014 -0.14949 -0.2585 -0.29038 1.0173 0.59803 -0.083486 0.30558 0.47593 0.026809 0.090965 0.052627 0.074359 -0.36702 0.20615')
        # motorbike
        gloves[14] = self.parse_vec('-0.71389 0.24032 0.18202 -0.088098 -0.0221 0.40635 -0.052222 -0.7371 0.10113 -0.53055 0.31645 0.038549 -0.19491 -0.32697 0.19943 -0.57279 0.37553 -0.032514 0.38359 0.17835 0.15913 -0.35313 0.02973 -0.23569 -0.43549 1.1884 0.14985 -0.16452 -0.40016 -0.56127 -0.33039 0.52782 0.3549 -0.20633 0.083044 0.2831 0.11659 0.30438 0.10226 0.10078 0.028741 0.47505 -0.20491 0.85168 0.27381 -0.36575 -0.10471 -0.12137 0.57409 0.20199 0.67158 0.24407 -0.37533 -0.47886 0.14464 0.39224 -0.0092274 0.1721 -0.1426 0.2096 0.29686 0.29672 0.51776 -0.47219 -0.47362 0.11739 -0.63042 0.49884 -0.23614 -0.24633 -0.70232 -0.25878 0.030875 -0.16369 0.46182 0.5136 -0.38601 -0.047488 -0.094195 0.13135 0.8398 0.36724 -0.71488 0.37607 -0.48174 0.091225 0.75595 0.025951 -0.53061 0.067239 -0.096866 -0.017189 -0.41604 -0.25577 0.062865 0.72158 0.40743 0.30805 -0.090951 0.41967 0.65751 -0.40103 -0.63055 0.71519 -0.21426 -0.93983 -0.062673 -0.26543 0.57424 -0.18904 -0.048883 0.16076 0.17014 0.066801 -0.33529 0.49848 -0.36642 -0.39713 -0.024494 0.20383 0.27226 -0.77433 0.30171 0.57367 -0.043116 0.079159 0.11836 0.44143 0.38227 0.052925 0.42209 0.25846 0.22436 0.099898 -0.56815 0.5277 0.045656 0.26299 -0.28591 -0.083705 -0.15414 0.31572 -0.33542 0.55337 -0.41729 0.14364 0.27906 0.25342 0.21754 0.048804 0.49028 0.074632 -0.86813 0.46902 0.071533 0.19098 0.034001 0.085827 0.53565 -0.41446 0.89643 0.55033 0.12035 0.35773 0.46279 -0.32748 -0.19938 -0.19853 -0.67121 -0.28168 0.062604 -0.08126 0.18431 -0.20818 0.5847 0.28582 0.29446 -0.4769 -0.039078 0.049103 0.16808 0.10381 0.10851 -0.38339 -0.52841 -0.53238 0.19347 -0.58147 -0.075303 0.94578 0.027874 0.1047 -0.28123 0.39113 -0.014863 -0.14572 0.27597 0.57036 0.051002 0.5199')
        # person
        gloves[15] = self.parse_vec('-0.0050341 0.43759 -0.10728 -0.12754 0.14574 0.44772 0.95882 -0.064739 -0.50419 0.33734 -0.023299 -0.16157 -0.50659 -0.19574 0.11752 0.45953 0.59953 0.52383 0.30061 -0.1844 0.13675 0.65594 -0.074337 -0.3523 -0.052698 1.6318 -0.0046084 -0.25087 0.089844 -0.18572 -0.22642 -0.10869 0.048051 -0.17346 -0.43151 0.046666 -0.17714 0.088511 0.2762 0.63112 0.41748 0.0931 0.13658 0.28507 -0.32909 0.089497 0.83896 0.098229 -0.059272 0.2835 -0.27827 0.19624 -0.049926 -0.69574 0.05352 0.060065 -0.068556 0.35591 -0.33751 -0.29361 -0.20059 -0.70989 0.46549 -0.44908 0.39502 0.49783 0.11653 0.54268 -0.48819 0.33826 -0.19704 -0.25727 0.26366 -0.22318 0.89299 -0.31712 0.10259 0.22438 -0.3718 -0.40868 0.38256 0.42004 -0.45121 -0.21513 0.014042 -0.049652 -0.11214 0.011164 -0.3606 -0.22827 -0.29906 0.53176 -0.054389 0.4932 -0.0052785 -0.086764 0.018286 -0.37717 0.51306 0.02191 0.014376 -0.40826 -0.054018 -0.92469 0.62715 -0.089945 0.20125 0.35328 -0.11475 0.15953 -0.26962 0.32959 0.060915 -0.14037 -0.20202 -0.2143 -0.034605 -0.011244 -0.59668 -0.091056 -0.71178 0.042869 -0.57287 -0.32826 -0.067884 -0.17087 -0.19935 -0.1571 0.044163 -0.31392 -0.23472 0.22923 -0.014186 0.6537 0.30681 -0.13804 0.021964 0.024048 0.47967 -0.3507 0.086764 0.68457 0.05042 -0.058323 0.59401 0.44433 -0.26444 -0.29732 0.031588 -0.43998 -0.16777 0.069608 -5.1022 0.47442 -0.27831 -0.10934 0.46917 -0.083847 0.25815 0.17722 0.39479 0.17018 -0.44708 -0.1237 -0.26057 -0.73399 -0.6979 0.36218 0.16067 -0.19531 0.13494 -0.14111 -0.2051 0.29239 -0.053072 0.0051988 -0.062671 -0.45236 0.38349 0.13699 -0.041298 0.29428 -0.23263 -0.032635 0.18313 0.23076 -0.62433 0.53785 0.33477 -0.2688 0.41107 -0.079753 0.32565 -0.42345 -0.12034 0.55607 -0.030407 0.2565 0.057437 -0.43445')
        # plant (instead of pottedplant)
        gloves[16] = self.parse_vec('-0.11111 0.057649 0.10509 -0.20679 -0.6105 0.31852 0.42001 -0.17437 0.082972 -0.54518 -0.39878 0.39278 -1.3354 -0.074956 -0.084963 0.30613 0.2464 -0.30133 -0.01795 -0.66445 -0.36118 -0.070666 -0.55449 0.11218 -0.2291 0.86695 -0.22428 0.14198 0.17671 -0.037329 0.037456 0.35369 0.29038 -0.65732 -0.20003 -0.2671 -0.053406 1.121 -0.17424 0.14477 -0.072602 -0.033538 -0.50043 0.23116 0.28005 0.62388 0.32728 -0.51442 0.1267 -0.31332 0.73642 0.3247 0.35872 -0.032586 0.022577 -0.04257 0.86809 0.13627 0.36609 0.28022 0.52616 0.79906 0.087693 -0.21913 0.024632 -0.053635 0.51211 0.17806 -0.40947 -0.079995 -0.56075 0.51136 0.77576 0.16721 0.19856 -0.00095677 -0.20017 -0.23092 0.37044 -0.58672 1.1888 0.084034 0.25076 0.077527 0.42798 0.18191 -0.15721 0.49148 -0.51208 0.1111 0.15223 0.25601 0.22023 -0.39595 -0.2301 0.51021 0.28086 0.5029 0.28635 0.40141 0.62646 0.091801 0.62058 -0.043011 -0.49427 -0.70087 -0.30576 -0.36211 -0.34107 0.57102 -0.8341 -0.070333 0.089223 -0.080423 -0.18906 -0.64046 -0.40274 0.22003 -1.3558 0.25769 -0.9221 0.25875 0.35121 -0.2247 0.10865 -0.44573 0.020137 0.42043 0.613 0.12949 0.36586 0.38093 -0.090628 0.62279 -0.20802 -1.1409 -0.07417 -0.103 -0.041084 -0.5689 -0.84172 -0.085305 -0.13616 -0.35128 -0.25108 -0.30743 -0.13205 0.0080276 0.17393 0.36135 0.18384 0.39011 -2.4754 -0.78184 -0.3437 0.1192 0.16951 0.10794 0.11712 -0.45028 -0.090958 0.19695 0.40981 -0.017894 0.36176 -0.23603 -0.38903 -0.1323 0.45368 -0.47899 -0.039468 -0.42611 0.72573 -0.39974 -0.44137 -0.30683 0.32104 0.75202 0.088467 -0.33332 -0.33403 -0.021241 -0.57665 -0.29877 0.86615 0.055897 0.44669 0.093072 0.36417 0.13264 0.26142 0.21439 1.0113 0.45556 -0.17741 0.63894 0.50394 -0.096641 0.093821 -0.37366')
        # sheep
        gloves[17] = self.parse_vec('-0.48883 -0.20854 0.19 0.47514 -0.51317 0.93146 0.39056 -0.063723 0.37478 -0.51601 0.17559 -0.43622 -0.43426 0.47602 0.37163 0.29038 -0.13016 -0.43486 0.46872 -0.37762 -0.27845 0.18054 0.19341 -0.34921 0.46227 1.0573 0.34446 -0.54068 0.025835 0.27329 0.0035218 0.72711 0.15759 0.23173 0.47177 0.091373 -0.33239 0.64459 0.2321 0.53376 0.77982 0.25495 0.32031 1.3828 0.32023 -0.28634 -0.034268 -0.29433 1.1474 0.54237 -0.6161 -0.40534 0.14573 -0.27524 0.19085 -0.27628 0.04799 0.4885 0.39575 0.36233 0.0032923 0.65519 -0.1076 0.089766 -0.4876 0.26514 -0.58268 0.032213 -0.090252 0.52496 0.10102 0.11129 -0.44439 -0.13691 -0.26121 0.0056128 -0.29664 -0.37598 0.39873 0.66627 0.97159 0.82369 -0.28087 1.0091 -0.50606 0.43705 0.14394 -0.11277 -0.11075 0.049597 0.11112 0.033074 -0.42926 -0.18468 -0.57982 -0.31848 0.59124 0.38171 0.18173 0.24726 0.33712 0.70201 -0.10992 0.79551 -0.34354 -0.32717 0.030538 0.24604 -0.16857 -0.77267 -0.45843 -0.0060385 -0.33472 -0.26437 -0.21247 0.24241 -0.46285 0.32434 0.077569 0.28511 -0.38589 -0.0041081 -0.19887 -0.50601 0.54081 -0.33611 -0.10492 -0.55035 0.66215 0.056054 0.0033579 0.51826 0.1167 0.49053 0.012476 -0.024986 -0.099266 0.069926 -0.50376 -0.26692 -0.52158 0.74391 -0.29793 -0.74214 0.13901 0.32073 -0.30176 0.19213 -0.071006 -0.47931 -0.045606 0.083413 -2.2626 -0.62771 0.18383 -0.20006 0.20747 0.93842 0.039725 -0.073622 0.95418 0.13252 0.27428 0.020871 -0.99478 -0.60409 0.17769 0.30552 -1.1167 0.1562 0.042333 0.60667 0.31924 0.026862 -0.10894 -0.059597 -0.37697 -0.12174 0.6681 -0.45187 -0.65058 0.26808 0.014851 0.25176 0.15575 0.14464 -0.050168 -0.16096 -0.52247 -0.58604 0.2535 -0.62094 -0.24828 0.26319 -0.77109 -0.49111 -0.56805 -0.14462 -1.0704 -0.44761')
        # sofa
        gloves[18] = self.parse_vec('-0.55126 -0.25437 0.40944 -0.37035 0.15619 0.74781 -0.11626 -0.089062 0.32154 -0.44794 -0.79694 0.2678 -0.28714 -0.18248 -0.099868 0.0044547 0.14694 -0.35398 -0.27681 0.50209 -0.69361 -0.18524 -0.24016 0.065474 -0.12418 -0.62691 0.10098 0.40514 -0.49283 -0.2973 -0.15004 -0.3799 0.16639 -0.27232 0.012692 -0.57008 0.71212 0.014504 -0.32036 0.36146 -0.24907 0.0045155 0.60505 0.10751 -0.012905 -0.7139 -0.0033347 0.49277 0.20195 0.24275 -0.4409 -0.85957 0.20849 -0.15209 -0.14172 0.089232 -0.01787 0.43873 0.50058 0.22154 -0.033405 0.13686 0.7646 -0.69327 -0.26288 -0.23971 -0.17341 -0.23787 0.43504 0.11938 0.11955 0.34672 -0.71679 -0.29109 -0.10691 0.68078 0.40884 0.11318 -0.57474 0.75584 1.0335 -0.2474 -0.15966 -0.012803 0.54782 0.057099 0.4468 -0.71945 0.045912 0.4424 0.12564 0.35803 -0.0099155 0.31759 0.24738 -0.64752 0.25505 -0.67181 -1.0412 1.119 0.64015 -0.11319 0.046755 -0.47351 -0.98552 -0.17908 -0.49623 -0.41556 -0.20261 -0.69094 -0.47285 0.26574 0.7891 -0.069767 -0.03166 -0.41765 0.44728 -0.89895 0.077562 0.1578 0.19055 0.31422 -0.034844 -0.0092608 0.52623 0.92685 0.31479 0.097895 -0.46618 -0.029873 0.22965 0.29097 -0.12395 -0.33712 -0.54815 -0.35642 0.058376 0.47008 0.11739 0.64565 0.26692 0.43143 -0.14349 -0.63512 0.016551 0.35399 -0.43784 -0.15109 0.20562 0.51676 0.31307 0.10661 -2.6111 0.43525 0.29244 0.30252 0.44164 -0.086565 0.36163 -0.42814 0.39952 0.6095 0.37047 0.41566 -0.097301 0.16248 0.10507 0.59692 0.20441 0.14534 0.3668 0.76102 -0.086382 0.24676 0.84287 -0.22111 -0.34975 0.40324 -0.2311 0.21589 0.12531 -0.19429 0.36596 0.18071 -0.090265 0.02224 -0.63616 0.52052 0.49612 0.082192 0.79162 -0.90827 0.53091 -0.42354 0.48783 -0.71943 -0.2491 0.33456 -0.13367 0.41854')
        # train
        gloves[19] = self.parse_vec('0.37548 -0.16669 0.20334 -0.1707 0.057389 0.63362 0.098189 0.17951 0.094536 0.61758 -0.012194 0.03028 -0.59888 -0.46359 -0.86279 0.16698 0.17168 0.33183 -0.34339 0.28135 0.25715 -0.61989 0.25431 -0.3545 -0.23358 0.82254 -0.19874 -0.59826 0.41849 -0.2918 -0.010124 0.035356 -0.22821 -0.024697 0.29794 -0.19534 -0.57675 0.1217 1.1021 -0.36827 -0.20924 -0.33711 -0.043826 -0.59845 0.2646 -0.51695 -0.33889 -0.12732 0.15502 -0.07516 0.34644 -0.75462 0.068238 -0.13422 -0.76469 0.37285 -0.052013 0.65885 -0.042933 -0.28987 -0.11953 0.083422 0.32609 -0.17798 -0.41476 -0.65127 0.44529 0.89459 -0.020621 -0.2502 -0.098399 0.38612 -0.090363 -0.42287 -0.031872 0.56521 -0.2458 0.25975 -0.40278 -0.15071 0.2289 -0.61254 -0.10832 0.045791 -0.082635 0.85964 -0.099326 0.072384 0.61234 -0.067309 0.44315 0.37082 -0.70074 1.0807 0.071388 0.44729 0.30407 0.2371 -0.085221 0.15809 0.75598 -0.35196 0.044777 -0.12434 -0.24014 -0.53403 0.24857 -0.171 1.1383 0.13646 -0.097531 -0.21034 0.08839 -0.52547 0.48343 -0.34049 0.08137 -0.54899 0.11817 1.1512 -0.078584 -0.3733 -0.15421 0.30997 -0.79899 -0.029586 -0.018026 -0.0035351 0.18488 1.0739 -0.055238 -0.14807 0.61702 -0.25605 -0.14685 -0.061761 -0.37885 0.479 -0.3825 -0.061271 -1.0717 0.37763 -0.74767 -0.40958 0.76752 -0.43954 0.2279 -0.42838 -0.80615 -1.0569 0.36154 -0.6756 -3.9798 0.43417 0.058424 -0.25163 0.017483 -0.03925 0.078241 0.47291 0.21551 0.32782 -0.19112 0.47168 -0.48036 -0.62983 -0.23916 -0.078116 -0.99057 -0.31946 -0.040178 -0.061123 -0.5638 -1.2017 1.0233 -0.81923 0.70827 0.47827 0.090528 0.32272 0.44516 0.26923 0.19288 0.69647 0.22837 -0.64528 0.13395 0.5601 0.58335 -0.065198 -0.016235 -0.18649 0.47786 0.54648 0.61327 0.14863 -0.098438 -0.33517 0.48419 0.22443')
        # monitor (instead of tvmonitor)
        gloves[20] = self.parse_vec('0.39506 0.57035 -0.34469 -0.2418 -0.085844 0.076654 -0.101 -0.043672 0.35994 -0.068255 0.2001 -0.18981 -0.807 -0.10697 -0.49271 -0.62257 0.033404 0.043097 -0.16137 0.037069 -0.092297 0.71918 -0.33535 0.99444 -0.23735 -0.18001 -0.3563 -0.035004 -0.42524 0.020921 0.59765 -0.68987 0.42215 -0.039423 0.81596 -1.0068 0.056338 -0.28865 -0.3757 0.41928 0.026622 0.43745 -0.34303 -0.038377 -0.74057 -0.060697 -0.25378 0.020666 -0.29184 -0.3001 0.0055599 0.24966 -0.58941 -0.46169 -0.14104 0.056481 -0.22584 -0.093435 0.50993 0.079872 0.085146 -0.030725 0.92953 0.31664 0.45899 -0.16236 0.18509 -0.3883 0.36874 -0.094179 0.080235 0.3334 -0.55517 -0.52172 0.035944 0.14773 0.20172 -0.43234 0.26623 -0.18526 0.447 0.72035 0.45101 -0.44633 0.42394 -0.31974 0.26068 0.39124 0.30794 0.27531 -0.74552 0.38866 0.24196 -0.015859 0.10625 -0.27747 0.19079 -0.52362 -0.65494 0.78146 0.40401 0.28761 0.43292 -0.73568 -0.43771 -0.6939 -0.48592 0.055997 0.086118 -0.63828 -0.23677 -0.0023004 0.41527 -0.12113 0.76192 0.55498 -0.13573 0.069332 -0.56071 0.57345 0.1751 0.061245 -0.19107 0.35412 0.44524 0.52874 -0.067264 0.26821 0.12174 0.060287 -0.47326 0.10187 0.044595 0.60027 -0.22113 -0.63658 -0.36007 -0.14078 -0.15648 -0.064581 0.19529 -0.47165 0.1875 -0.20738 0.49041 -0.15263 -0.18486 -0.62451 -0.065947 -0.18168 0.55734 -0.3354 -2.1831 0.073034 -0.55689 0.20047 0.48988 -0.25795 0.16526 0.13197 -0.0024545 0.0057711 0.74506 0.007249 -0.53246 0.045723 -0.45975 -0.96999 0.74073 0.1641 0.49533 -0.030726 0.11014 -0.36607 0.04882 -0.26971 0.52963 0.40551 -0.31313 0.26866 0.19646 0.71257 0.22745 -0.50536 0.34653 0.51053 0.014612 -0.1723 0.056945 0.66266 0.71526 0.03419 0.17104 -0.049182 -0.27842 -0.29963 0.41816 0.16741 0.34322 0.25798')
        return gloves

    def set_emb_vecs(self, emb_selection):
        if emb_selection is None:
            return None
        elif emb_selection == 'glove':
            embs = self.set_glove()
        elif emb_selection == 'random':
            embs = np.random.randn(len(self.CLASSES), self.EMB_DIM)
        elif emb_selection == 'farthest_points':
            pts = np.random.randn(100 * len(self.CLASSES), self.EMB_DIM)
            inds = generate_farthest_vecs(pts, len(self.CLASSES))
            embs = pts[inds]
        elif emb_selection == 'orthogonal':
            pts = np.random.randn(self.EMB_DIM, self.EMB_DIM)
            q, _ = np.linalg.qr(pts, 'complete')
            embs = q.T[:len(self.CLASSES)]
        else:
            raise AssertionError('Unknown emb_selection: {}'.format(emb_selection))
        embs = embs.astype(np.float32)
        return embs

    # def prepare_test_img(self, idx):
    #     img_info = self.img_infos[idx]
    #     ann_info = self.get_ann_info(idx)
    #     results = dict(img_info=img_info, ann_info=ann_info)
    #     self.pre_pipeline(results)
    #     seg_map = self.gt_seg_map_loader(results)
    #     results = self.pipeline(results)
    #     results['gt_semantic_seg'] = seg_map['gt_semantic_seg']
    #     return results
