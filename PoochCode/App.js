import React from "react";
import "./scss/main.scss";

//Navigation
import NavBar from "./components/NavBar";
import NavBarLoggedOut from "./components/NavBarLoggedOut";

// Router
import { Router } from "@reach/router";

//Firebase
import fire from "./config/Fire";

// Components
import Footer from "./components/Footer";

// Pages
import ChangePassword from "./pages/ChangePassword";
import Home from "./pages/Home";
import Profile from "./pages/Profile";
import YourPets from "./pages/YourPets";
import DogWalking from "./pages/DogWalking";
import DogBoarding from "./pages/DogBoarding";
import DogMeetup from "./pages/DogMeetup";
import DogServices from "./pages/DogServices";
import DogRecords from "./pages/DogRecords";
import PageNotFound from "./pages/PageNotFound";

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      user: {}
    };
  }

  componentDidMount() {
    this.authListener();
  }

  authListener() {
    fire.auth().onAuthStateChanged(user => {
      if (user) {
        this.setState({ user });
      } else {
        this.setState({ user: null });
      }
    });
  }

  render() {
    return (
      <div className="App">
        {this.state.user ? (
          <div>
            <NavBar />
            <Router>
              <Home path="/" />
              <Profile path="/profile" />
              <YourPets path="/your-pets" />
              <DogWalking path="/dog-walking" />
              <DogBoarding path="/dog-boarding" />
              <DogMeetup path="/dog-meetup" />
              <DogServices path="/dog-services" />
              <DogRecords path="/dog-records" />
              <PageNotFound path="/page-not-found" default />
            </Router>
          </div>
        ) : (
          <div>
            <NavBarLoggedOut />
            <Router>
              <Home path="/" />
              <PageNotFound path="/page-not-found" default />
            </Router>
          </div>
        )}
        <Footer/>
      </div>
    );
  }
}

export default App;
