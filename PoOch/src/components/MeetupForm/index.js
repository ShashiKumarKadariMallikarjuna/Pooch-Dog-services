import React from "react";
import Fire from "../../config/Fire.js"; // Firebase
import "./styles.scss"; // Styles
import Loader from "react-loader-spinner"; // Loader
import DogPark from "./dogpark.png"
import Close from "./close.jpg";
import AddButton from "./add-button.png";

class MeetupForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      timestamp: "",
      address: "",
      city: "",
      sta: "",
      zipcode: "",
      description: "",
      search: "",
      startDate: "",
      endDate: ""
    };

    this.handleChange = this.handleChange.bind(this);
    this.updateSearch = this.updateSearch.bind(this);
  }

  handleChange(e) {
    this.setState({ [e.target.name]: e.target.value });
  }

  updateSearch(e) { this.setState({search: e.target.value.substr(0,50) }); }

  componentDidMount() {
    document.getElementById("error-message").style.display = "none";
    // Set search start date to today
    var today = new Date();
    var month = (today.getMonth()+1).toString();
    if (month.length == 1)
      month = "0"+month;
    var day = today.getDate().toString();
    if (day.length == 1)
      day = "0"+day;
    document.getElementById("start-date").value = today.getFullYear()+"-"+month+"-"+day;
  }

  componentDidUpdate() {
    document.getElementById("loader").style.display = "block";
    document.getElementById("meetup-bubble-home").innerHTML = null; // Clear old data before updating
    
    // Get meetups from Firebase
    Fire.firestore().collection("meetups")
      .orderBy(document.getElementById("meetup-search-category").value).get()
      .then(snapshot => {
        snapshot.docs.forEach(doc => {
          renderMeetup(doc);
        });
      });

    function renderMeetup(doc) {
      // Bubble render
      var newBox = document.createElement("div");
      newBox.className = "meetup-box row mx-3 pooch-body";
      //newBox.addEventListener("click", ()=>{viewMeetup(doc.data().date, doc.data().time, doc.data().address)});
      var leftCol = document.createElement("div");
      leftCol.className = "box-left-col";
      newBox.appendChild(leftCol);

      // Left column
      var newPic = document.createElement("img");
      newPic.className = "box-meetup-pic";
      newPic.style.left = "0%";
      newPic.style.marginTop = (leftCol.scrollHeight/4 - newPic.height/2).toString()+"px";
      newPic.src = doc.data().pic;
      newPic.alt = "Meetup Picture";
      leftCol.appendChild(newPic);

      // Right column
      var rightCol = document.createElement("div");
      rightCol.className = "box-right-col";
      newBox.appendChild(rightCol);
      var nameDiv = document.createElement("div");
      rightCol.appendChild(nameDiv);

      // Make title camelcase
      var titleWords = doc.data().title.toLowerCase().split(" ");
      var adjustedTitle = "";
      for (var i=0; i<titleWords.length; i++)
        adjustedTitle += titleWords[i].substr(0,1).toUpperCase() + titleWords[i].substr(1, titleWords[i].length-1) + " ";
      nameDiv.textContent = adjustedTitle+" by ";

      nameDiv.className = "box-row";
      nameDiv.style.fontWeight = "600";
      var nameStart = "";
      Fire.firestore().collection("parents").where("email", "==", doc.data().email).get()
      .then(snapshot => {
        snapshot.forEach(doc => {
          nameStart = doc.data().name;
        })
      });
      Fire.firestore().collection("boarders").where("email", "==", doc.data().email).get()
      .then(snapshot => {
        snapshot.forEach(doc => {
          nameStart = doc.data().name;
        })
      });
      Fire.firestore().collection("boarders").where("email", "==", doc.data().email).get()
      .then(snapshot => {
        snapshot.forEach(doc => {
          nameStart = doc.data().name;
        })
      });
      setTimeout(() => {
        // Make name camelcase (must wait for firebase to return)
        var nameWords = nameStart.toLowerCase().split(" ");
        var adjustedName = "";
        for (var i=0; i<nameWords.length; i++)
          adjustedName += nameWords[i].substr(0,1).toUpperCase() + nameWords[i].substr(1, nameWords[i].length-1) + " ";
        nameDiv.textContent += adjustedName;
      }, 500);
      
      var address = document.createElement("div");
      address.className = "box-row";
      // Make address camelcase
      var addressWords = doc.data().address.toLowerCase().split(" ");
      var adjustedAddress = "";
      for (var i=0; i<addressWords.length; i++)
        adjustedAddress += addressWords[i].substr(0,1).toUpperCase() + addressWords[i].substr(1, addressWords[i].length-1) + " ";
      address.textContent = adjustedAddress;
      rightCol.appendChild(address);

      var city = document.createElement("div");
      city.className = "box-row";
      // Make city camelcase
      var cityWords = doc.data().city.toLowerCase().split(" ");
      var adjustedCity = "";
      for (var i=0; i<cityWords.length; i++)
        adjustedCity += cityWords[i].substr(0,1).toUpperCase() + cityWords[i].substr(1, cityWords[i].length-1) + " ";
      city.textContent = adjustedCity+", "+doc.data().state+", "+doc.data().zipcode;
      rightCol.appendChild(city);

      var date = document.createElement("div");
      date.className = "box-row";
      var months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
      var adjustedMonth = months[parseInt(doc.data().date.substr(5,2))-1];
      var adjustedTime;
      if (parseInt(doc.data().time.substr(0,2)) < 12)
        adjustedTime = doc.data().time.substr(1,1)+doc.data().time.substr(2,3)+"AM";
      else if (parseInt(doc.data().time.substr(0,2)) == 12)
        adjustedTime = doc.data().time.substr(0,2)+doc.data().time.substr(2,3)+"PM";
      else
        adjustedTime = (parseInt(doc.data().time.substr(0,2))-12).toString()+doc.data().time.substr(2,3)+"PM";
      date.textContent = doc.data().date.substr(8,2)+" "+adjustedMonth+" "+doc.data().date.substr(0,4)+" @ "+adjustedTime;
      var formattedDate = doc.data().date.substr(0,4)+"-"+doc.data().date.substr(5,2)+"-"+doc.data().date.substr(8,2);
      rightCol.appendChild(date);

      var buttonRow = document.createElement("div");
      buttonRow.style.width = "100%";
      buttonRow.style.fontSize = "70%";
      buttonRow.style.marginTop = "2%";
      newBox.appendChild(buttonRow);

      var maybeButton = document.createElement("button");
      maybeButton.textContent = "Maybe";
      maybeButton.style.marginLeft = "15%";
      maybeButton.style.width = "25%";
      maybeButton.style.marginRight = "10%";
      maybeButton.style.borderRadius = "10px";
      maybeButton.style.backgroundColor = "rgb(245, 245, 245)";
      maybeButton.style.color = "black";
      Fire.firestore().collection("maybe") // Check if maybe
      .where("email", "==", Fire.auth().currentUser.email)
      .where("meetupID", "==", doc.id).get()
      .then(snapshot => {
        snapshot.forEach(doc => {
          maybeButton.style.backgroundColor = "rgb(65, 65, 65)";
          maybeButton.style.color = "white";
        })
      });
      maybeButton.addEventListener("click", ()=>{toggleMaybe(maybeButton, yesButton, doc.id)});
      buttonRow.appendChild(maybeButton);

      var yesButton = document.createElement("button");
      yesButton.textContent = "I'm Going";
      yesButton.style.marginLeft = "10%";
      yesButton.style.width = "25%";
      yesButton.style.marginRight = "15%";
      yesButton.style.borderRadius = "10px";
      yesButton.style.backgroundColor = "rgb(245, 245, 245)";
      yesButton.style.color = "black";
      Fire.firestore().collection("going") // Check if going
      .where("email", "==", Fire.auth().currentUser.email)
      .where("meetupID", "==", doc.id).get()
      .then(snapshot => {
        snapshot.forEach(doc => {
          yesButton.style.backgroundColor = "rgb(65, 65, 65)";
          yesButton.style.color = "white";
        })
      });
      yesButton.addEventListener("click", ()=>{toggleYes(maybeButton, yesButton, doc.id)});
      buttonRow.appendChild(yesButton);

      // Search functionality
      var matchSearch = true;
      var searchCategory = document.getElementById("meetup-search-category").value;
      var searchBar = document.getElementById("meetup-search").value;
      if (searchCategory === "title") { // Searching by title
        if (searchBar.length > nameDiv.textContent.split("by")[0].length) // Check if search bar is longer than title
          matchSearch = false;
        else
          for (var j=0; j<searchBar.length; j++) {// Loop through each letter in search bar
            if (searchBar[j].toLowerCase() !== nameDiv.textContent.split("by")[0][j].toLowerCase())
              matchSearch = false;
          }
      }
      if (searchCategory === "city") { // Searching by city
        if (searchBar.length > city.textContent.split(", ")[0].length) // Check if search bar is longer than city
          matchSearch = false;
        else
          for (var j=0; j<searchBar.length; j++) // Loop through each letter in search bar
            if (searchBar[j].toLowerCase() !== city.textContent.split(", ")[0][j].toLowerCase())
              matchSearch = false;
      }
      if (searchCategory === "zipcode") { // Searching by zipcode
        if (searchBar.length > city.textContent.split(", ")[2].length) // Check if search bar is longer than zipcode
          matchSearch = false;
        else
          for (var j=0; j<searchBar.length; j++) // Loop through each letter in search bar
            if (searchBar[j].toLowerCase() !== city.textContent.split(", ")[2][j].toLowerCase())
              matchSearch = false;
      }
      if (searchCategory === "address") { // Searching by address
        if (searchBar.length > address.textContent.length) // Check if search bar is longer than address
          matchSearch = false;
        else
          for (var j=0; j<searchBar.length; j++) // Loop through each letter in search bar
            if (searchBar[j].toLowerCase() !== address.textContent[j].toLowerCase())
              matchSearch = false;
      }
      // Make sure meetup is equal or later than search start date
      var startDate = document.getElementById("start-date").value;
      // If meetup year before search year => no match
      if (parseInt(formattedDate.substr(0,4)) < parseInt(startDate.substr(0,4))) { 
        matchSearch = false;
        console.log("Year too small: "+formattedDate+" < "+startDate);
      }
      // If meetup year equal to search year, but meetup month earlier than search month => no match
      else if (parseInt(formattedDate.substr(0,4)) === parseInt(startDate.substr(0,4)) && parseInt(formattedDate.substr(5,2)) < parseInt(startDate.substr(5,2))) { 
        matchSearch = false;
        console.log("Month too small: "+formattedDate+" < "+startDate);
      }
      // If meetup year equal to search year and meetup month equal to search month, but meetup day earlier than search day => no match
      else if ( (parseInt(formattedDate.substr(0,4)) === parseInt(startDate.substr(0,4))) && (parseInt(formattedDate.substr(5,2)) === parseInt(startDate.substr(5,2))) && (parseInt(formattedDate.substr(8,2)) < parseInt(startDate.substr(8,2))) ) {
        matchSearch = false;
        console.log("Day too small: "+formattedDate+" < "+startDate);
      }
      //Make sure meetup is equal or earlier than search end date
      var endDate = document.getElementById("end-date").value;
      // If meetup year after search year => no match
      if (parseInt(formattedDate.substr(0,4)) > parseInt(endDate.substr(0,4))) { 
        matchSearch = false;
        console.log("Year too big: "+formattedDate+" > "+endDate);
      }
      // If meetup year equal to search year, but meetup month after search month => no match
      else if (parseInt(formattedDate.substr(0,4)) === parseInt(endDate.substr(0,4)) && parseInt(formattedDate.substr(5,2)) > parseInt(endDate.substr(5,2))) { 
        matchSearch = false;
        console.log("Month too big: "+formattedDate+" > "+endDate);
      }
      // If meetup year equal to search year and meetup month equal to search month, but meetup day after search day => no match
      else if ( (parseInt(formattedDate.substr(0,4)) === parseInt(endDate.substr(0,4))) && (parseInt(formattedDate.substr(5,2)) === parseInt(endDate.substr(5,2))) && (parseInt(formattedDate.substr(8,2)) > parseInt(endDate.substr(8,2))) ) {
        matchSearch = false;
        console.log("Day too big: "+formattedDate+" > "+endDate);
      }
      setTimeout(() => {
        // Only add bubble if it matches search
        if (matchSearch === true)
          document.getElementById("meetup-bubble-home").appendChild(newBox);
        document.getElementById("loader").style.display = "none";
      }, 250);
    }
    
    function toggleYes(maybeButton, yesButton, docID) { 
      maybeButton.style.backgroundColor = "rgb(245, 245, 245)";
      maybeButton.style.color = "black";
      if (yesButton.style.backgroundColor === "rgb(245, 245, 245)") {
        yesButton.style.backgroundColor = "rgb(65, 65, 65)";
        yesButton.style.color = "white";
        Fire.firestore().collection("maybe") // Delete from maybe
      .where("email", "==", Fire.auth().currentUser.email)
      .where("meetupID", "==", docID).get()
      .then(snapshot => {
        snapshot.forEach(doc => {
          doc.ref.delete();
        })
      });
      Fire.firestore().collection("going") // Delete from going
        .where("email", "==", Fire.auth().currentUser.email)
        .where("meetupID", "==", docID).get()
        .then(snapshot => {
          snapshot.forEach(doc => {
            doc.ref.delete();
          })
        });
        setTimeout(() => {
          Fire.firestore().collection("going").add({ // Add to going
            email: Fire.auth().currentUser.email,
            meetupID: docID
          });
        }, 500);
      
      }
      else {
        yesButton.style.backgroundColor = "rgb(245, 245, 245)";
        yesButton.style.color = "black";
        Fire.firestore().collection("going") // Delete from going
        .where("email", "==", Fire.auth().currentUser.email)
        .where("meetupID", "==", docID).get()
        .then(snapshot => {
          snapshot.forEach(doc => {
            doc.ref.delete();
          })
        });
      }
    }

    function toggleMaybe(maybeButton, yesButton, docID) { 
      yesButton.style.backgroundColor = "rgb(245, 245, 245)";
      yesButton.style.color = "black";
      if (maybeButton.style.backgroundColor === "rgb(245, 245, 245)") {
      maybeButton.style.backgroundColor = "rgb(65, 65, 65)";
      maybeButton.style.color = "white";
      Fire.firestore().collection("going") // Delete from going
      .where("email", "==", Fire.auth().currentUser.email)
      .where("meetupID", "==", docID).get()
      .then(snapshot => {
        snapshot.forEach(doc => {
          doc.ref.delete();
        })
      });
      Fire.firestore().collection("maybe") // Delete from maybe
      .where("email", "==", Fire.auth().currentUser.email)
      .where("meetupID", "==", docID).get()
      .then(snapshot => {
        snapshot.forEach(doc => {
          doc.ref.delete();
        })
      });
      setTimeout(() => {
        Fire.firestore().collection("maybe").add({ // Add to maybe
          email: Fire.auth().currentUser.email,
          meetupID: docID
        });
      }, 500);
      
    }
    else {
      maybeButton.style.backgroundColor = "rgb(245, 245, 245)";
      maybeButton.style.color = "black";
      Fire.firestore().collection("maybe") // Delete from maybe
      .where("email", "==", Fire.auth().currentUser.email)
      .where("meetupID", "==", docID).get()
      .then(snapshot => {
        snapshot.forEach(doc => {
          doc.ref.delete();
        })
      });
    }
    }
  }

  submitMeetup() {
      Fire.firestore().collection("meetups").where('email', '==', Fire.auth().currentUser.email)
      .where("address", "==", document.getElementById("new-meetup-address").value)
      .where("date", "==", document.getElementById("new-meetup-date").value)
      .where("time", "==", document.getElementById("new-meetup-time").value)
      .get().then(function(querySnapshot) {
        querySnapshot.forEach(function(doc) {
          doc.ref.delete();
        });
      });
      setTimeout(() => {
        Fire.firestore().collection("meetups").add({
          email: Fire.auth().currentUser.email,
          pic: document.getElementById("new-meetup-pic").src,
          title: document.getElementById("new-meetup-title").value.toLowerCase(),
          address: document.getElementById("new-meetup-address").value.toLowerCase(),
          city: document.getElementById("new-meetup-city").value.toLowerCase(),
          zipcode: document.getElementById("new-meetup-zipcode").value,
          state: document.getElementById("new-meetup-state").value.toUpperCase(),
          date: document.getElementById("new-meetup-date").value,
          time: document.getElementById("new-meetup-time").value
      }).catch(error => {
      document.getElementById("error-message").style.display = "block";
    });
    }, 1000);
      setTimeout(() => {
        if (document.getElementById("error-message").style.display !== "block")
          window.location.reload();
      }, 2000);
  }

  previewPic(event) {
    var input = event.target; // Get image that changed state of input element
    var reader = new FileReader();
    reader.onload = function() {
    document.getElementById(input.id.split("-input")[0]+"-pic").src = reader.result;
    };
    reader.readAsDataURL(input.files[0]); // Show preview of image
  }

  openNewMeetup() { document.getElementById("meetup-popup").className = "collapse.show" }

  closeNewMeetup() { document.getElementById("meetup-popup").className = "collapse" }

  closeError() { document.getElementById("error-message").style.display = "none" }

  viewMeetupsThis(duration) {
    var today = new Date();
    if (duration === "week") {
      var month = getFormattedMonth(today);
      var day = getFormattedDay(today);
      document.getElementById("start-date").value = today.getFullYear()+"-"+month+"-"+day;
      today.setDate(today.getDate()+7);
      month = getFormattedMonth(today);
      day = getFormattedDay(today);
      document.getElementById("end-date").value = today.getFullYear()+"-"+month+"-"+day;
      // Highlight selection
      document.getElementById("week-block").style.backgroundColor = "rgb(65,65,65)";
      document.getElementById("week-block").style.color = "white";
      document.getElementById("weekend-block").style.backgroundColor = "rgb(220,220,220)";
      document.getElementById("weekend-block").style.color = "black";
      document.getElementById("two-week-block").style.backgroundColor = "rgb(220,220,220)";
      document.getElementById("two-week-block").style.color = "black";
      document.getElementById("month-block").style.backgroundColor = "rgb(220,220,220)";
      document.getElementById("month-block").style.color = "black";
      this.setState({startDate: document.getElementById("start-date").value});
    }
    else if (duration === "weekend") {
      var daysUntilFriday = 0;
      if (today.getDay() < 5)
        daysUntilFriday = 5-today.getDay();
      else if (today.getDay() == 5)
        daysUntilFriday = 7;
      else
      daysUntilFriday = 6;
      today.setDate(today.getDate()+daysUntilFriday);
      var month = getFormattedMonth(today);
      var day = getFormattedDay(today);
      document.getElementById("start-date").value = (today.getFullYear()+"-"+month+"-"+day);
      today.setDate(today.getDate()+2);
      month = getFormattedMonth(today);
      day = getFormattedDay(today);
      // Highlight selection
      document.getElementById("end-date").value = (today.getFullYear()+"-"+month+"-"+day);
      document.getElementById("week-block").style.backgroundColor = "rgb(220,220,220)";
      document.getElementById("week-block").style.color = "black";
      document.getElementById("weekend-block").style.backgroundColor = "rgb(65,65,65)";
      document.getElementById("weekend-block").style.color = "white";
      document.getElementById("two-week-block").style.backgroundColor = "rgb(220,220,220)";
      document.getElementById("two-week-block").style.color = "black";
      document.getElementById("month-block").style.backgroundColor = "rgb(220,220,220)";
      document.getElementById("month-block").style.color = "black";
      this.setState({startDate: document.getElementById("start-date").value});
    }
    else if (duration === "two weeks") {
      var month = getFormattedMonth(today);
      var day = getFormattedDay(today);
      document.getElementById("start-date").value = today.getFullYear()+"-"+month+"-"+day;
      today.setDate(today.getDate()+14);
      month = getFormattedMonth(today);
      day = getFormattedDay(today);
      document.getElementById("end-date").value = today.getFullYear()+"-"+month+"-"+day;
      // Highlight selection
      document.getElementById("end-date").value = (today.getFullYear()+"-"+month+"-"+day);
      document.getElementById("week-block").style.backgroundColor = "rgb(220,220,220)";
      document.getElementById("week-block").style.color = "black";
      document.getElementById("weekend-block").style.backgroundColor = "rgb(220,220,220)";
      document.getElementById("weekend-block").style.color = "black";
      document.getElementById("two-week-block").style.backgroundColor = "rgb(65,65,65)";
      document.getElementById("two-week-block").style.color = "white";
      document.getElementById("month-block").style.backgroundColor = "rgb(220,220,220)";
      document.getElementById("month-block").style.color = "black";
      this.setState({startDate: document.getElementById("start-date").value});
    }
    else if (duration === "month") {
      var month = getFormattedMonth(today);
      var day = getFormattedDay(today);
      document.getElementById("start-date").value = today.getFullYear()+"-"+month+"-"+day;
      today.setMonth(today.getMonth()+1);
      today.setDate(1);
      today.setDate(today.getDate()-1);
      month = getFormattedMonth(today);
      day = getFormattedDay(today);
      document.getElementById("end-date").value = today.getFullYear()+"-"+month+"-"+day;
      // Highlight selection
      document.getElementById("end-date").value = (today.getFullYear()+"-"+month+"-"+day);
      document.getElementById("week-block").style.backgroundColor = "rgb(220,220,220)";
      document.getElementById("week-block").style.color = "black";
      document.getElementById("weekend-block").style.backgroundColor = "rgb(220,220,220)";
      document.getElementById("weekend-block").style.color = "black";
      document.getElementById("two-week-block").style.backgroundColor = "rgb(220,220,220)";
      document.getElementById("two-week-block").style.color = "black";
      document.getElementById("month-block").style.backgroundColor = "rgb(65,65,65)";
      document.getElementById("month-block").style.color = "white";
      this.setState({startDate: document.getElementById("start-date").value});
    }
    
    function getFormattedDay(today) {
      var day = today.getDate().toString();
      if (day.length == 1)
        day = "0"+day;
      return day;
    }

    function getFormattedMonth(today) {
      var month = (today.getMonth()+1).toString();
      if (month.length == 1)
        month = "0"+month;
      return month;
    }
  }

  maybeGoing() {
    Fire.firestore().collection("meetups").add({
      maybe: Fire.auth().currentUser.email
  });
  }

  render() {
    return (
      <div id="meetup-form">
        <div className="row meetup-search-row">
          <div style={{paddingTop: "0.5%"}}>From</div><input name="startDate" className="date-input" id="start-date" type="date" min="2020-01-01" max="2030-01-01" onChange={this.handleChange}/>
          <div style={{paddingTop: "0.5%"}}>To</div><input name="endDate" className="date-input" id="end-date" type="date" min="document.getElementById('start-date').value" max="2030-01-01" onChange={this.handleChange}/>
          <input className="meetup-search-bar" id="meetup-search" placeholder="Search..." maxLength="50" onChange={this.updateSearch} value={this.state.search}></input>
          <select className="meetup-search-dropdown" id="meetup-search-category" onChange={this.handleChange}>
            <option value="title">Title</option>
            <option value="address">Address</option>
            <option value="city">City</option>
            <option value="zipcode">Zipcode</option>
          </select>
        </div>
        <div className="row" id="time-row">
          <div className="time-block" id="week-block" onClick={()=>this.viewMeetupsThis("week")}>This week</div>
          <div className="time-block" id="weekend-block" onClick={()=>this.viewMeetupsThis("weekend")}>This weekend</div>
          <div className="time-block" id="two-week-block" onClick={()=>this.viewMeetupsThis("two weeks")}>Next two weeks</div>
          <div className="time-block" id="month-block" onClick={()=>this.viewMeetupsThis("month")}>This month</div>
        </div>
        <div className="pooch-title">Dog Meetups<img id="add-dog-button" alt="add-meetup-button" src={AddButton} onClick={this.openNewMeetup} style={{width:"3.5vw", marginTop: "0%"}}/></div>
        <div id="loader" className="mb-4"><Loader type="ThreeDots" color="black" height={75} width={75}/></div>
        <div id="meetup-bubble-home" className="row mb-8"></div>

        <div className="collapse" id="meetup-popup">
          <div className="row">
            <div className="box-left-col">
              <img className="popup-meetup-pic" id="new-meetup-pic" src={DogPark} alt="New Meetup Picture"/> 
            </div>
            <div className="box-right-col">
              <img src={Close} className="pull-right" id="close-popup-button" onClick={this.closeNewMeetup}/>
              <input className="box-row" id="new-meetup-title" placeholder="Title" maxLength="16"/>
              <input className="box-row" id="new-meetup-address" placeholder="Address" maxLength="32"/>
              <input className="box-row" id="new-meetup-city" placeholder="City" maxLength="32"/>
              <input className="box-row" id="new-meetup-state" placeholder="State" maxLength="2"/>
              <input className="box-row" id="new-meetup-zipcode" placeholder="Zipcode" maxLength="5"/>
              <input className="box-row" id="new-meetup-date" type="date" min="2020-01-01" max="2030-01-01"/>
              <input className="box-row" id="new-meetup-time" type="time" min="5:00" max="20:00"/>
            </div>
          </div>
          <input className="row" type="file" id="new-meetup-input" onChange={this.previewPic}/>
          <button id="submit-meetup" className="btn my-2" onClick={this.submitMeetup}>Submit Meetup</button>
        </div>
        <div className="fixed-top" id="error-message">
          <div>Image size too large</div>
          <button className="m2-3 mr-4 btn-danger popup-button" onClick={this.closeError}>Close</button>
        </div>
      </div>
    );
  }
}

export default MeetupForm;
 