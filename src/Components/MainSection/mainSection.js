import React from "react";

import "./mainSection.css";
import Button from "../../UIElements/Button/button";

const MainSection = () => {
  return (
    <React.Fragment>
      <div className="allSections">
        <div className="mainSection">
          <div className="content">
            <div className="mainName">
              Transform your body, <br /> with Tech.
            </div>
            <div className="heroSubtitle">
              Unleash Your Strength: <br /> Perfect Your Form with Video-Based
              Gym Correction
            </div>
            <div className="aboutSection">
              Video-Based Exercise Form Detection improves performance by
              providing real-time feedback to athletes, enthusiasts and fitness
              apps. Advanced computer vision techniques analyze exercise form,
              helping to optimize workouts and cultivate healthy habits.
            </div>
            <a href="/exercises">
              <Button title="Try it now" />
            </a>
          </div>
        </div>
        <div className="code">DO It.</div>
      </div>
    </React.Fragment>
  );
};

export default MainSection;
