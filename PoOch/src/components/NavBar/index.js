import React from "react";
import Fire from "../../config/Fire"; // Firebase
import "./styles.scss"; // Styles
import GenericProfile from "./generic-profile.png";
import MessageTab from "../MessageTab";

class NavBar extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      dropdownOpen: false,
    };
    this.logout = this.logout.bind(this);
  }

  componentDidMount() {
    
    Fire.auth().onAuthStateChanged(function(user) {
      if (user) { // User is signed in
        Fire.firestore().collection("profile-types") // Check if user has profile picture
        .where("email", "==", Fire.auth().currentUser.email)
        .get()
        .then(snapshot => {
          snapshot.docs.forEach(doc => {
            if (doc.data().type === "parent")
            renderParentPic();
          if (doc.data().type === "walker")
            renderWalkerPic();
          if (doc.data().type === "boarder")
            renderBoarderPic();
          });
        });
      }
    });
    function renderParentPic(doc) {
      Fire.firestore().collection("parents")
      .where("email", "==", Fire.auth().currentUser.email)
      .get()
      .then(snapshot => {
        snapshot.docs.forEach(doc => {
        document.getElementById("profile-pic").src = doc.data().pic; 
        });
      });
    }
    function renderWalkerPic(doc) {
      Fire.firestore().collection("walkers")
      .where("email", "==", Fire.auth().currentUser.email)
      .get()
      .then(snapshot => {
        snapshot.docs.forEach(doc => {
        document.getElementById("profile-pic").src = doc.data().pic; 
        });
      });
    }
    function renderBoarderPic(doc) {
      Fire.firestore().collection("boarders")
      .where("email", "==", Fire.auth().currentUser.email)
      .get()
      .then(snapshot => {
        snapshot.docs.forEach(doc => {
        document.getElementById("profile-pic").src = doc.data().pic; 
        });
      });
    }
    // Bold current page
    if(window.location.pathname.substr(1, window.location.pathname.length-1) === "your-pets")
      document.getElementById("your-pets").style.fontWeight = "600";
    else if(window.location.pathname.substr(1, window.location.pathname.length-1) === "dog-walking")
      document.getElementById("dog-walking").style.fontWeight = "600";
    else if(window.location.pathname.substr(1, window.location.pathname.length-1) === "dog-boarding")
      document.getElementById("dog-boarding").style.fontWeight = "600";
    else if(window.location.pathname.substr(1, window.location.pathname.length-1) === "dog-services")
      document.getElementById("dog-services").style.fontWeight = "600";
    else if(window.location.pathname.substr(1, window.location.pathname.length-1) === "dog-meetup")
      document.getElementById("dog-meetup").style.fontWeight = "600";
    else if(window.location.pathname.substr(1, window.location.pathname.length-1) === "dog-records")
      document.getElementById("dog-records").style.fontWeight = "600";
  }

  logout(e) {
    e.preventDefault();
    Fire.auth().signOut();
    // Redirect to login page so user doesn't get 404 error
    setTimeout(() => {
      window.location.replace("/");
    }, 100);
  }

  goToProfile() {
    window.location.replace("/profile");
  }

  toggleCollapse() {
    if (document.getElementById("profile-dropdown").className === "collapse fixed-top") {
        document.getElementById("profile-dropdown").className = "collapse.show fixed-top";
    }
    else if (document.getElementById("profile-dropdown").className === "collapse.show fixed-top") {
        document.getElementById("profile-dropdown").className = "collapse fixed-top"; 
    }
  }

  collapseDropdown() {
        document.getElementById("profile-dropdown").className = "collapse fixed-top";
  }

  openDropdown() {
    document.getElementById("profile-dropdown").className = "collapse.show fixed-top";
  }

  openMessages() {
    if (document.getElementById("message-tab").className === "pl-5 collapse")
      document.getElementById("message-tab").className = "pl-5 collapse.show";
    else
      document.getElementById("message-tab").className = "pl-5 collapse";
  }

  render() {
    return (
      <div onMouseLeave={this.collapseDropdown}>
        <MessageTab/>
        <div className="collapse fixed-top" id="profile-dropdown" >
            <div className="pooch-navbar-item row" id="dropdown-button" onClick={this.goToProfile}>📷 Profile</div>
            {/* <div className="pooch-navbar-item row" id="dropdown-button" onClick={this.openMessages}>✉️ Messages</div> */}
            <div className="pooch-navbar-item row" id="dropdown-button" onClick={this.logout}>🔌 Logout</div>
        </div>
        <nav className="nav-bar row fixed-top py-5" >
          <div id="nav-bar-pic"><img id="profile-pic" src={GenericProfile} alt="Profile" onClick={this.goToProfile} onMouseEnter={this.openDropdown}/></div>
          <a className="pooch-brand" id="pooch"  href="/">POOCH</a>
          <div className="pooch-navbar-item row" id="nav-pages">
                <a className="col" id="your-pets" href="/your-pets">Your Pets</a>
                <a className="col" id="dog-walking" href="/dog-walking">Dog Walking</a>
                <a className="col" id="dog-boarding" href="/dog-boarding">Dog Boarding</a>
                <a className="col" id="dog-services" href="/dog-services">Dog Services</a>
                <a className="col" id="dog-meetup" href="/dog-meetup">Dog Meetups</a>
                <a className="col" id="dog-records" href="/dog-records">Dog Records</a> 
          </div>
        </nav>     
      </div>
    );
  }
}

export default NavBar;
