import React from "react";

import "./mainSide.css";

const MainSide = () => {
  return (
    <React.Fragment>
      <div className="sideContainer">
        <img
          className="sideImage"
          src={require("../../Images/dumbbells-g68f0268d9_1920.jpg")}
        />
      </div>
    </React.Fragment>
  );
};

export default MainSide;
