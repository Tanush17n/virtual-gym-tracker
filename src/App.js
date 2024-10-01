import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";

import "./App.css";
import FrontPage from "./Pages/FrontPage/frontPage";
import ExercisePage from './Pages/ExercisePage/exercisePage';
import MediaPipePage from "./Pages/MediaPipePage/mediaPipePage";
import NavBar from "./Components/NavBar/navBar";
import Footer from "./Components/Footer/footer";

const App = () => {
  return (
    <Router>
      <NavBar />
      <Routes>
        <Route exact path="/" element={<FrontPage />} />
        <Route exact path="/exercises" element={<ExercisePage />} />
        <Route exact path="/mediaPipeOutput" element={<MediaPipePage />} />
      </Routes>
      <Footer />
    </Router>
  );
};

export default App;
