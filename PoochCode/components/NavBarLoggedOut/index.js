// React
import React from "react";

// Styles
import "./styles.scss";

class NavBarLoggedOut extends React.Component {

  openSignup() {
    document.getElementById("login-box").className = "collapse";
    document.getElementById("password-box").className = "collapse";
    document.getElementById("signup-box").className = "collapse.show";
  }

  openLogin() {
    document.getElementById("signup-box").className = "collapse";
    document.getElementById("password-box").className = "collapse";
    document.getElementById("login-box").className = "collapse.show";
  }

  render() {
    return (
      <div className="navbar">
        <div className="collapse fixed-top" id="profile-dropdown">
            <div className="pooch-navbar-item mb-3" id="dropdown-row"><button id="dropdown-button" onClick={this.goToProfile}>Profile</button></div>
            <div className="pooch-navbar-item mt-3" id="dropdown-row"><button id="dropdown-button" onClick={this.logout}>Logout</button></div>
        </div>
        <div className="pooch-navbar-item fixed-top" id="new-tip">New to Pooch? Start here <b>↑</b></div>
        <nav className="nav-bar row fixed-top py-5" >
          <a className="pooch-brand" id="pooch"  href="/">POOCH</a>
          <div className="pooch-navbar-item row" id="nav-pages">
            <a classname="col" id="your-pets" href="/">Your Pets</a>
            <a classname="col" id="dog-walking" href="/">Dog Walking</a>
            <a classname="col" id="dog-boarding" href="/">Dog Boarding</a>
            <a classname="col" id="dog-services" href="/">Dog Services</a>
            <a classname="col" id="dog-meetup" href="/">Dog Meetups</a>
            <a classname="col" id="dog-records" href="/">Dog Records</a> 
          </div>
          <div className="col pooch-navbar-item mr-5" id="profile-dropdown">
            <div className="row mb-3" id="dropdown-item" onClick={this.openLogin}>Login</div>
            <div className="row mt-3" id="dropdown-item" onClick={this.openSignup}>Create Account</div>
          </div>
        </nav>          
      </div>
    );
  }
}

export default NavBarLoggedOut;
