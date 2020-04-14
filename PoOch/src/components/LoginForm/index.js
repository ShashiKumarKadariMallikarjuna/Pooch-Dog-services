// React
import React from "react";

// Firebase
import fire, {googleProvider} from "../../config/Fire";

import "./styles.scss";

class LoginForm extends React.Component {
  constructor(props) {
    super(props);

    this.login = this.login.bind(this);
  }

  login(e) {
    e.preventDefault();
    fire
      .auth()
      .signInWithEmailAndPassword(document.getElementById("login-email").value, document.getElementById("login-password").value)
      .catch(error => {
        document.querySelector("#error-message-login").style.display = "block";
        document.getElementById("error-message-login").innerHTML = error.message;
      });
      setTimeout(() => {
        fire.auth().onAuthStateChanged(user => {
            if (user) {
                document.getElementById("login-box").className = "collapse";
                window.location.replace("/profile"); 
            } else {
                console.log("Error: User could not be logged in due to bad credentials");
            }
        });
    }, 500);
  }

  googleLogin(e) {
    e.preventDefault();
    fire.auth().signInWithPopup(googleProvider);
    setTimeout(() => {
        fire.auth().onAuthStateChanged(user => {
            if (user) {
                document.getElementById("login-box").className = "collapse";
                window.location.replace("/profile"); 
            } else {
                console.log("Error: User could not be logged in due to bad credentials");
            }
        });
    }, 1000);
  }

  closeLogin() {
    document.getElementById("login-box").className = "collapse"
  }

  resetPassword() {
    document.getElementById("login-box").className = "collapse";
    document.getElementById("signup-box").className = "collapse";
    document.getElementById("password-box").className = "collapse.show";
  }

  render() {
    return (
      <div id="login-box" className="collapse">
          <div className="pooch-navbar-item login-title">Login to your Account</div>
          <input id="login-email" type="text" className="login-input" placeholder="Email"/>
          <input id="login-password" type="password" className="login-input" placeholder="Password" maxlength="50"/>
          <div className="row login-button-row">
            <div id="login-button" onClick={this.login}>Login</div>
            <div id="google-button" onClick={this.googleLogin}>Login with Google</div> 
          </div>
          <div id="close-button" onClick={this.resetPassword}>Forgot your password?</div>
          <div id="close-button" onClick={this.closeLogin}>Not what you wanted? Click here to close</div>  
          <div id="error-message-login"/>
      </div>
    );
  }
}

export default LoginForm;
