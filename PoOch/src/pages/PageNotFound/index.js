import React from "react";
import Fade from "react-reveal"; // Animation

class PageNotFound extends React.Component {
  render() {
    return (
    <div className="my-7">
      <Fade>
        <div className="pooch-title my-5">Woops!</div>
        <div className="pooch-subtitle my-5">Page not found. Click the button below to go back home</div>
        <a className="pooch-navbar-item" href="/">Go Home</a>
      </Fade>
    </div>
    );
  }
}

export default PageNotFound;
