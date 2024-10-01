import React from "react";

import "./button.css";

const Button = (props) => {
  return <React.Fragment>
    <button className="allButtons" onClick={props.onClickHandler}>
        {props.title}
    </button>
  </React.Fragment>;
};

export default Button;
