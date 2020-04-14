import React from "react";
import "./styles.scss"; // Styles
import Slide from "react-reveal";
import Generic from "../NavBar/generic-profile.png";

class MessageTab extends React.Component {
constructor(props) {
    super(props);
    this.state = {
        name: "",
        phone: ""
    };
this.handleChange = this.handleChange.bind(this);
}

handleChange(e) { this.setState({ [e.target.name]: e.target.value }); }

render() {
  return (
      <Slide right>
        <div id="message-tab" className="pl-5 collapse">
        <div>Friends</div>
        <div className="row">
            <img className="friend-pic" src={Generic} alt="Friend Profile Picture"></img>
            Doug Walker
        </div>
        <div className="row">Sally Sue</div>
        <div className="row">Mickey Mouse</div>
        <div className="row">George Jefferson</div>
        <div>Converation with Doug Walker</div>
        <div className="row pull-left">Hi, how are you?</div>
        <div className="row pull-right">Good</div>
        <div className="row pull-left">Anything new?</div>
        <div className="row pull-right">No</div>
        </div>
    </Slide>
  );
}


}

export default MessageTab;