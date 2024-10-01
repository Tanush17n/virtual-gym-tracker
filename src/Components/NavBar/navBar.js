import React from "react";

import "./navBar.css";
import Button from "../../UIElements/Button/button";

const NavBar = () => {
  const onClickHandler = () => {};

  return (
    <React.Fragment>
      <header className="navBar">
        <div className="navContents">
          <div className="nineDotsSection">
            {/* <NineDots /> */}
            <a className="topIcon" href="/">
              <img
                className="navBarIcon"
                src={require("../../Images/wired-outline-1764-pushups .gif")}
              />
            </a>
          </div>
          <div className="rightMost">
            <a>
              <img
                className="colorSwitchIcon"
                src={require("../../Images/sun.png")}
              />
            </a>
            <Button title="Connect" onClickHandler={onClickHandler} />
          </div>
        </div>
      </header>
    </React.Fragment>
  );
};

export default NavBar;
