import React from "react";

import "./thirdCenter.css";

const SecondMain = () => {
  return (
    <React.Fragment>
      <div className="wholeThird">
        <div className="thirdHeader">Benefits</div>
        <div className="thirdMain">
          <div className="thirdProject">
            <div className="projectHeadings">Precision</div>
            Video processing form correction allows you to achieve perfect form
            by analyzing your exercise movements in real-time, ensuring that you
            are performing each exercise correctly and safely.
          </div>
          <div className="thirdProject">
            <div className="projectHeadings">Feedback</div>
            With video processing form correction, you receive instant feedback
            on your performance, helping you identify areas that need
            improvement and allowing you to make adjustments on the fly.
          </div>
          <div className="thirdProject">
            <div className="projectHeadings">Progress tracking</div>
            tracking Video processing form correction technology can also track
            your progress over time, allowing you to see how far you've come and
            motivating you to keep pushing towards your fitness goals.
          </div>
        </div>
      </div>
    </React.Fragment>
  );
};

export default SecondMain;
