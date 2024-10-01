import React from "react";

import "./frontPage.css";
import MainSection from "../../Components/MainSection/mainSection";
import MainSide from "../../Components/MainSide/mainSide";
import SecondMain from '../../Components/SecondMain/secondMain';
import SecondSide from '../../Components/SecondSide/secondSide';
import ThirdCenter from '../../Components/ThirdCenter/thirdCenter';

const FrontPage = () => {
  return (
    <React.Fragment>
      <div className="frontPage">
        <MainSection />
        <MainSide />
      </div>
      <div className="secondPage">
        <SecondSide />
        <SecondMain />
      </div>
      <div className="thirdPage">
        <ThirdCenter />
      </div>
    </React.Fragment>
  );
};

export default FrontPage;
