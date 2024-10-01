import React from "react";

import "./nineDots.css";

const NineDots = () => {
  return (
    <React.Fragment>
      <button className="nineButton">
        <div className="mainNineDots">
          <div className="dots fly-two"></div>
          <div className="dots fly-one"></div>
          <div className="dots fly-two"></div>
          <div className="dots fly-one"></div>
          <div className="dots fly-one"></div>
          <div className="dots fly-one"></div>
          <div className="dots fly-two"></div>
          <div className="dots fly-one"></div>
          <div className="dots fly-two"></div>
        </div>
      </button>
    </React.Fragment>
  );
};

export default NineDots;
