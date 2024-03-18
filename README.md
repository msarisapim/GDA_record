# Nutrition Recording System using OCR
Developed an innovative Nutrition Recording System aimed at streamlining dietary tracking by integrating YOLO (You Only Look Once) object detection with Optical Character Recognition (OCR). This system is designed to automatically identify and analyze nutritional information from food packaging, enabling users to effortlessly record and monitor their dietary intake.


# Yolo OBB
reference: https://docs.ultralytics.com/datasets/obb/


# YOLO OBB Format
The YOLO OBB format uses four normalized corner points to define bounding boxes, structured as:
class_index, x1, y1, x2, y2, x3, y3, x4, y4


# Directory structure
\documentclass{article}
\usepackage[edges]{forest}

\begin{document}

\begin{forest}
  for tree={
    grow=east,
    parent anchor=east, 
    child anchor=west,
    anchor=west,
    draw,
    align=center,
    edge path={
      \noexpand\path [draw, \forestoption{edge}] 
      (!u.parent anchor) -- +(5pt,0) |- (.child anchor)\forestoption{edge label};
    },
  }
[GDA-OBB
  [labels
    [val]
    [train]
  ]
  [images
    [val]
    [train]
  ]
]
\end{forest}

\end{document}

