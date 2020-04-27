import React from "react";
import RecordForm from "../../components/RecordForm"; // Inner component
import Fade from "react-reveal"; // Animation

class DogRecords extends React.Component {
  render(){
    return (
        <Fade> <RecordForm /> </Fade>
    );
  }
}

export default DogRecords;