// React
import React from "react";

// Firebase
import fire from "../../config/Fire";

import "./styles.scss";

class SignUpForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      email: "",
      password: "",
      confirmPassword: ""
    };
    this.signup = this.signup.bind(this);
    this.handleChange = this.handleChange.bind(this);
  }

  handleChange(e) {
    this.setState({ [e.target.name]: e.target.value });
  }

  signup(e) {
    e.preventDefault();
    if (this.state.password === this.state.confirmPassword) {
       fire
      .auth()
      .createUserWithEmailAndPassword(this.state.email, this.state.password)
      .catch(error => {
        document.querySelector("#error-message-signup").style.display = "block";
        document.getElementById("error-message-signup").innerHTML = error.message;
      }); 
      // If successful, then login
      setTimeout(() => {
        fire
      .auth()
      .signInWithEmailAndPassword(this.state.email, this.state.password)
      .catch(error => {
        document.querySelector("#error-message-login").style.display = "block";
        document.getElementById("error-message-login").innerHTML = error.message;
      });
      }, 500);
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
    else {
        document.querySelector("#error-message-signup").style.display = "block";
        document.getElementById("error-message-signup").innerHTML = "Passwords do not match";
    }
  }

  closeSignup() {
    document.getElementById("signup-box").className = "collapse"
  }

  render() {
    return (
      <div id="signup-box" className="collapse">
          <div className="pooch-navbar-item login-title">Create an Account</div>
          <input name="email" id="signup-email" type="text" className="login-input" placeholder="Email" maxlength="30" value={this.state.email} onChange={this.handleChange}/>
          <input name="password" id="signup-password" type="password" className="login-input" placeholder="Password" maxlength="30" value={this.state.password} onChange={this.handleChange}/>
          <input name="confirmPassword" id="signup-password-confirm" type="password" className="login-input" placeholder="Confirm Password" maxlength="30" value={this.state.confirmPassword} onChange={this.handleChange}/>
          <div className="row login-button-row">
            <div id="signup-button" onClick={this.signup}>Create Account</div>
          </div>
          <div id="close-button" onClick={this.closeSignup}>Not what you wanted? Click here to close</div>
          <div id="error-message-signup"></div>
      </div>
    );
  }
}

export default SignUpForm;
