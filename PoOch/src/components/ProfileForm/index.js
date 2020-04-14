import React from "react";
import Fire from "../../config/Fire.js"; // Firebase
import "./styles.scss"; // Styles
//Animation and Images
import GenericPic from "./generic-profile.png";
import Fade from "react-reveal";
import Loader from "react-loader-spinner";

class ProfileForm extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      email: "",
      currentPassword: "",
      newPassword: ""
    };

    this.handleChange = this.handleChange.bind(this);
  }

  handleChange(e) {
    this.setState({ [e.target.name]: e.target.value });
  }

  componentDidMount() {
    document.getElementById("error-message").style.display = "none";
    document.getElementById("upload-loader").style.display = "none";
    document.getElementById("profile-loader").style.display = "block";
    
    console.log("component mounted");
    setTimeout(() => { // Determine user type
      //document.getElementById("username").innerHTML = "Welcome, "+Fire.auth().currentUser.email;
      Fire.firestore().collection("profile-types")
      .where("email", "==", Fire.auth().currentUser.email)
      .get()
      .then(snapshot => {
        snapshot.docs.forEach(doc => {
          console.log("checking if parent");
          if (doc.data().type === "parent")
            renderParentProfile();
            console.log("checking if walker");
          if (doc.data().type === "walker")
            renderWalkerProfile();
            console.log("checking if boarder");
          if (doc.data().type === "boarder")
            renderBoarderProfile();
        });
      });
    }, 1000);
    setTimeout(() => { // Wait 1.5 seconds, then hide loader
          document.getElementById("profile-loader").style.display = "none";
      }, 1500);
    setTimeout(() => { // Wait 2.25 seconds, then check if no profile visible, so do first time setup
    if (document.getElementById("parent-profile").className === "react-reveal collapse" && 
        document.getElementById("walker-profile").className === "react-reveal collapse" &&
        document.getElementById("boarder-profile").className === "react-reveal collapse") {
          console.log("user has not set up a profile!");
          document.getElementById("parent-or-provider").className = "collapse.show";
        }
    }, 2250);

        function renderParentProfile(doc) {
          console.log("user is a parent!");
          Fire.firestore().collection("parents")
          .where("email", "==", Fire.auth().currentUser.email)
          .get()
          .then(snapshot => {
            snapshot.docs.forEach(doc => {
            document.getElementById("parent-profile").className = "react-reveal collapse.show";
            // Make name camelcase
            var nameWords = doc.data().name.toLowerCase().split(" ");
            var adjustedName = "";
            for (var i=0; i<nameWords.length; i++)
              adjustedName += nameWords[i].substr(0,1).toUpperCase() + nameWords[i].substr(1, nameWords[i].length-1) + " ";
            document.getElementById("database-parent-name").value = adjustedName;
            document.getElementById("database-parent-phone").value = doc.data().phone;
            // Make city camelcase
            var cityWords = doc.data().city.toLowerCase().split(" ");
            var adjustedCity = "";
            for (var i=0; i<cityWords.length; i++)
              adjustedCity += cityWords[i].substr(0,1).toUpperCase() + cityWords[i].substr(1, cityWords[i].length-1) + " ";
            document.getElementById("database-parent-city").value = adjustedCity;
            document.getElementById("database-parent-pic").src = doc.data().pic;
            document.getElementById("delete-button-row").className = "collapse.show";
            });
          });
        }
      
        function renderWalkerProfile(doc) {
          console.log("user is a walker!");
          Fire.firestore().collection("walkers")
          .where("email", "==", Fire.auth().currentUser.email)
          .get()
          .then(snapshot => {
            snapshot.docs.forEach(doc => {
          document.getElementById("walker-profile").className = "react-reveal collapse.show";
          // Make name camelcase
          var nameWords = doc.data().name.toLowerCase().split(" ");
          var adjustedName = "";
          for (var i=0; i<nameWords.length; i++)
            adjustedName += nameWords[i].substr(0,1).toUpperCase() + nameWords[i].substr(1, nameWords[i].length-1) + " ";
          document.getElementById("database-walker-name").value = adjustedName;
          document.getElementById("database-walker-phone").value = doc.data().phone;
          // Make city camelcase
          var cityWords = doc.data().city.toLowerCase().split(" ");
          var adjustedCity = "";
          for (var i=0; i<cityWords.length; i++)
            adjustedCity += cityWords[i].substr(0,1).toUpperCase() + cityWords[i].substr(1, cityWords[i].length-1) + " ";
          document.getElementById("database-walker-city").value = adjustedCity;
          document.getElementById("database-walker-hourly-rate").value = doc.data().hourlyRate;
          document.getElementById("database-walker-pic").src = doc.data().pic;
          document.getElementById("delete-button-row").className = "collapse.show";
            });
          });
        }
      
        function renderBoarderProfile(doc) {
          console.log("user is a boarder!");
          Fire.firestore().collection("boarders")
          .where("email", "==", Fire.auth().currentUser.email)
          .get()
          .then(snapshot => {
            snapshot.docs.forEach(doc => {
          document.getElementById("boarder-profile").className = "react-reveal collapse.show";
          // Make name camelcase
          var nameWords = doc.data().name.toLowerCase().split(" ");
          var adjustedName = "";
          for (var i=0; i<nameWords.length; i++)
            adjustedName += nameWords[i].substr(0,1).toUpperCase() + nameWords[i].substr(1, nameWords[i].length-1) + " ";
          document.getElementById("database-boarder-name").value = adjustedName;
          document.getElementById("database-boarder-phone").value = doc.data().phone;
          // Make address camelcase
          var addressWords = doc.data().address.toLowerCase().split(" ");
          var adjustedAddress = "";
          for (var i=0; i<addressWords.length; i++)
            adjustedAddress += addressWords[i].substr(0,1).toUpperCase() + addressWords[i].substr(1, addressWords[i].length-1) + " ";
          document.getElementById("database-boarder-address").value = adjustedAddress;
          // Make city camelcase
          var cityWords = doc.data().city.toLowerCase().split(" ");
          var adjustedCity = "";
          for (var i=0; i<cityWords.length; i++)
            adjustedCity += cityWords[i].substr(0,1).toUpperCase() + cityWords[i].substr(1, cityWords[i].length-1) + " ";
          document.getElementById("database-boarder-city").value = adjustedCity;
          document.getElementById("database-boarder-daily-rate").value = doc.data().dailyRate;
          document.getElementById("database-boarder-pic").src = doc.data().pic;
          document.getElementById("delete-button-row").className = "collapse.show";
            });
          });
        }
  }

  startParentProfile() {
    document.getElementById("parent-button").style.backgroundColor="gray"; // Start parent profile
    document.getElementById("new-parent-profile").className="collapse.show";
    document.getElementById("service-button").style.backgroundColor="#f2f4f7"; // Close potentially open service profile
    document.getElementById("walker-or-boarder").className="collapse";
    document.getElementById("new-walker-profile").className="collapse";
    document.getElementById("new-boarder-profile").className="collapse";
  }

  startServiceProfile() {
    document.getElementById("service-button").style.backgroundColor="gray"; // Start service profile
    document.getElementById("walker-or-boarder").className="collapse.show";
    document.getElementById("parent-button").style.backgroundColor="#f2f4f7"; // Close potentially open parent profile
    document.getElementById("walker-button").style.backgroundColor="#f2f4f7";
    document.getElementById("boarder-button").style.backgroundColor="#f2f4f7";
    document.getElementById("new-parent-profile").className="collapse";
    document.getElementById("new-walker-profile").className="collapse";
    document.getElementById("new-boarder-profile").className="collapse";
  }

  startWalkerProfile() {
    document.getElementById("walker-button").style.backgroundColor="gray"; // Start walker profile
    document.getElementById("new-walker-profile").className="collapse.show";
    document.getElementById("boarder-button").style.backgroundColor="#f2f4f7"; // Close potentially open boarder profile
    document.getElementById("new-boarder-profile").className="collapse";
  }

  startBoarderProfile() {
    document.getElementById("boarder-button").style.backgroundColor="gray"; // Start boarder profile
    document.getElementById("new-boarder-profile").className="collapse.show";
    document.getElementById("walker-button").style.backgroundColor="#f2f4f7"; // Close potentially open walker profile
    document.getElementById("new-walker-profile").className="collapse";
  }

  updateParent() {
    
    Fire.firestore().collection("parents")
      .where('email', '==', Fire.auth().currentUser.email) // Check if parent exists
      .get()
      .then(snapshot => {
        if (snapshot.empty) { // Create new parent
          Fire.firestore().collection("parents").add({
            email: Fire.auth().currentUser.email,
            name: document.getElementById("parent-name").value,
            phone: document.getElementById("parent-phone").value,
            city: document.getElementById("parent-city").value,
            pic: document.getElementById("parent-pic").src
          })
          .catch(error => {
            document.getElementById("error-message").style.display = "block";
          });
          setTimeout(() => { // Only reload page if no error
            if (document.getElementById("error-message").style.display === "none") {
              Fire.firestore().collection("profile-types").add({
                email: Fire.auth().currentUser.email,
                type: "parent"
              });
            setTimeout(() => { window.location.reload(); }, 500);
            }
          }, 2000);
        }
        else { // Already exists, so update
          var oldPic;
          snapshot.forEach(doc => {
            oldPic = doc.data().pic; // Save old profile picture in case new one is too large
            doc.ref.delete(); // Delete old entries
            })
            Fire.firestore().collection("parents").add({
              email: Fire.auth().currentUser.email,
              name: document.getElementById("database-parent-name").value,
              phone: document.getElementById("database-parent-phone").value,
              city: document.getElementById("database-parent-city").value,
              pic: document.getElementById("database-parent-pic").src
            })
            .catch(error => {
              document.getElementById("error-message").style.display = "block";
              Fire.firestore().collection("parents").add({ // Try adding without the picture
                email: Fire.auth().currentUser.email,
                name: document.getElementById("database-parent-name").value,
                phone: document.getElementById("database-parent-phone").value,
                pic: oldPic
              })
            });
            setTimeout(() => { // Only reload page if no error
              if (document.getElementById("error-message").style.display === "none")
                window.location.reload();
            }, 2000);
        }
      });
  }

  updateWalker() {
    
    Fire.firestore().collection("walkers")
      .where('email', '==', Fire.auth().currentUser.email) // Check if walker exists
      .get()
      .then(snapshot => {
        if (snapshot.empty) { // Create new walker
          Fire.firestore().collection("walkers").add({
            email: Fire.auth().currentUser.email,
            name: document.getElementById("walker-name").value,
            phone: document.getElementById("walker-phone").value,
            city: document.getElementById("walker-city").value,
            houryRate: document.getElementById("walker-hourly-rate").value,
            pic: document.getElementById("walker-pic").src
          })
          .catch(error => {
            document.getElementById("error-message").style.display = "block";
          });
          setTimeout(() => { // Only reload page if no error
            if (document.getElementById("error-message").style.display === "none") {
              Fire.firestore().collection("profile-types").add({
                email: Fire.auth().currentUser.email,
                type: "walker"
              });
            setTimeout(() => { window.location.reload(); }, 500);
            }
          }, 2000);
        }
        else { // Already exists, so update
          var oldPic;
          snapshot.forEach(doc => {
            oldPic = doc.data().pic; // Save old profile picture in case new one is too large
            doc.ref.delete(); // Delete old entries
            })
            Fire.firestore().collection("walkers").add({
              email: Fire.auth().currentUser.email,
              name: document.getElementById("database-walker-name").value,
              phone: document.getElementById("database-walker-phone").value,
              city: document.getElementById("database-walker-city").value,
              houryRate: document.getElementById("database-walker-houry-rate").value,
              pic: document.getElementById("database-walker-pic").src
            })
            .catch(error => {
              document.getElementById("error-message").style.display = "block";
              Fire.firestore().collection("walkers").add({ // Try adding without the picture
                email: Fire.auth().currentUser.email,
                name: document.getElementById("database-walker-name").value,
                phone: document.getElementById("database-walker-phone").value,
                address: document.getElementById("database-walker-address").value,
                city: document.getElementById("database-walker-city").value,
                dailyRate: document.getElementById("database-walker-daily-rate").value,
                pic: oldPic
              })
            });
            setTimeout(() => { // Only reload page if no error
              if (document.getElementById("error-message").style.display === "none")
                window.location.reload();
            }, 2000);
        }
      });
  }

  updateBoarder() {
    
    Fire.firestore().collection("boarders")
      .where('email', '==', Fire.auth().currentUser.email) // Check if boarder exists
      .get()
      .then(snapshot => {
        if (snapshot.empty) { // Create new boarder
          Fire.firestore().collection("boarders").add({
            email: Fire.auth().currentUser.email,
            name: document.getElementById("boarder-name").value,
            phone: document.getElementById("boarder-phone").value,
            address: document.getElementById("boarder-address").value,
            city: document.getElementById("boarder-city").value,
            dailyRate: document.getElementById("boarder-daily-rate").value,
            pic: document.getElementById("boarder-pic").src
          })
          .catch(error => {
            document.getElementById("error-message").style.display = "block";
          });
          setTimeout(() => { // Only reload page if no error
            if (document.getElementById("error-message").style.display === "none") {
              Fire.firestore().collection("profile-types").add({
                email: Fire.auth().currentUser.email,
                type: "boarder"
              });
            setTimeout(() => { window.location.reload(); }, 500);
            }
          }, 2000);
        }
        else { // Already exists, so update
          var oldPic;
          snapshot.forEach(doc => {
            oldPic = doc.data().pic; // Save old profile picture in case new one is too large
            doc.ref.delete(); // Delete old entries
            })
            Fire.firestore().collection("boarders").add({
              email: Fire.auth().currentUser.email,
              name: document.getElementById("database-boarder-name").value,
              phone: document.getElementById("database-boarder-phone").value,
              address: document.getElementById("database-boarder-address").value,
              city: document.getElementById("database-boarder-city").value,
              dailyRate: document.getElementById("database-boarder-daily-rate").value,
              pic: document.getElementById("database-boarder-pic").src
            })
            .catch(error => {
              document.getElementById("error-message").style.display = "block";
              Fire.firestore().collection("boarders").add({ // Try adding without the picture
                email: Fire.auth().currentUser.email,
                name: document.getElementById("database-boarder-name").value,
                phone: document.getElementById("database-boarder-phone").value,
                address: document.getElementById("database-boarder-address").value,
                city: document.getElementById("database-boarder-city").value,
                dailyRate: document.getElementById("database-boarder-daily-rate").value,
                pic: oldPic
              })
            });
            setTimeout(() => { // Only reload page if no error
              if (document.getElementById("error-message").style.display === "none")
                window.location.reload();
            }, 2000);
        }
      });
  }

previewPic(event) {
  var input = event.target; // Get image that changed state of input element
  var reader = new FileReader();
  reader.onload = function() {
  document.getElementById(input.id.split("-input")[0]+"-pic").src = reader.result;
  };
  reader.readAsDataURL(input.files[0]); // Show preview of image
}

expand(event) {
  if(event.target.textContent === "Show More ▼") {
    document.getElementById(event.target.id.split("expand")[0]+"featured-images").className = "collapse.show";
    event.target.textContent = "Show More ▲";
  }
  else {
    document.getElementById(event.target.id.split("expand")[0]+"featured-images").className = "collapse";
    event.target.textContent = "Show More ▼";
  }
  
}

openDeletePopup() {document.getElementById("delete-popup").className="fixed-top collapse.show";}

closeDeletePopup() {document.getElementById("delete-popup").className="fixed-top collapse";}

deleteProfile() {
  Fire.firestore().collection("parents")
  .where('email', '==', Fire.auth().currentUser.email)
  .get().then(function(querySnapshot) {
      querySnapshot.forEach(function(doc) {
        doc.ref.delete();
      });
    });
  Fire.firestore().collection("walkers")
  .where('email', '==', Fire.auth().currentUser.email)
  .get().then(function(querySnapshot) {
      querySnapshot.forEach(function(doc) {
        doc.ref.delete();
      });
    });
  Fire.firestore().collection("boarders")
  .where('email', '==', Fire.auth().currentUser.email)
  .get().then(function(querySnapshot) {
      querySnapshot.forEach(function(doc) {
        doc.ref.delete();
      });
    });
  Fire.firestore().collection("profile-types")
  .where('email', '==', Fire.auth().currentUser.email)
  .get().then(function(querySnapshot) {
      querySnapshot.forEach(function(doc) {
        doc.ref.delete();
      });
    });
    setTimeout(() => {
      window.location.reload();
    }, 1000);
}

closeError() { document.getElementById("error-message").style.display = "none"; }

  render() {
    return (
      <div id="profile-form">
        <div id="profile-loader"><Loader type="TailSpin" color="black" height={150} width={150}/></div>
        <Fade>
          <div className="pooch-navbar-item mb-3" id="username"></div>

          <div id="parent-or-provider" className="collapse">
          <div className="my-4">Thank you for joining POOCH!</div>
          <div className="my-4">Help us get to know you by answering a few questions:</div>
          <div className="mt-6 mb-4">I am a...</div>
            <div className="row my-5 mx-8">
              <button className="col mx-4 py-1 signup" id="parent-button" onClick={this.startParentProfile}>
                Dog Parent
              </button>
              or a
              <button className="col mx-4 py-1 signup" id="service-button" onClick={this.startServiceProfile}>
                Dog Service Provider
              </button>
            </div>
          </div>
  
          <div id="new-parent-profile" className="collapse">
              <div className="mb-3"><img className="profile-pic" id="parent-pic" src={GenericPic} alt="Profile" onClick={this.toggleCollapse}/></div>
              <input type="file" id="parent-input" onChange={this.previewPic}/>
              <div className="col">
                <div className="my-4 row"><input className="profile-input" id="parent-name" type="text" placeholder="Name" maxLength="50"/></div>
                <div className="my-4 row"><input className="profile-input" id="parent-phone" type="text" placeholder="Phone Number" min="0" max="9999999999" maxLength="11"/></div>
                <div className="my-4 row"><input className="profile-input" id="parent-city" type="text" placeholder="City" maxLength="50"/></div> 
              </div>
              <button type="submit" className="btn btn-primary" onClick={this.updateParent}>Create Parent Profile</button>
          </div>

          <div id="walker-or-boarder" className="collapse">
          <div className="mt-6 mb-4">I would like to offer...</div>
            <div className="row my-5 mx-8">
              <button className="col mx-4 py-1 signup" id="walker-button" onClick={this.startWalkerProfile}>
                Dog Walking
              </button>
              or
              <button className="col mx-4 py-1 signup" id="boarder-button" onClick={this.startBoarderProfile}>
                Dog Boarding
              </button>
            </div>
          </div>
          
          <div id="new-walker-profile" className="collapse">
            <div className="mb-3"><img className="profile-pic" id="walker-pic" src={GenericPic} alt="Profile"/></div>
            <input type="file" id="walker-input" onChange={this.previewPic}/>
              <div className="col">
                <div className="my-4 row"><input className="profile-input" id="walker-name" type="text" placeholder="Name" maxLength="50"/></div>
                <div className="my-4 row"><input className="profile-input" id="walker-phone" type="text" placeholder="Phone Number" min="0" max="9999999999" maxLength="11"/></div>
                <div className="my-4 row"><input className="profile-input" id="walker-city" type="text" placeholder="City" maxLength="50"/></div>
                <div className="my-4 row"><input className="profile-input" id="walker-hourly-rate" type="text" placeholder="Hourly Rate" min="12" max="100" maxLength="3"/></div>
              </div>
              <button type="submit" className="btn btn-primary" onClick={this.updateWalker}>Create Dog Walking Profile</button>
          </div>

          <div id="new-boarder-profile" className="collapse">
            <div className="mb-3"><img className="profile-pic" id="boarder-pic" src={GenericPic} alt="Profile"/></div>
            <input type="file" id="boarder-input" onChange={this.previewPic}/>
              <div className="col">
                <div className="my-4 row"><input className="profile-input" id="boarder-name" type="text" placeholder="Name" maxLength="50"/></div>
                <div className="my-4 row"><input className="profile-input" id="boarder-phone" type="text" placeholder="Phone Number" min="0" max="9999999999" maxLength="11"/></div>
                <div className="my-4 row"><input className="profile-input" id="boarder-address" type="text" placeholder="Address" maxLength="50"/></div>
                <div className="my-4 row"><input className="profile-input" id="boarder-city" type="text" placeholder="City" maxLength="50"/></div>
                <div className="my-4 row"><input className="profile-input" id="boarder-daily-rate" type="text" placeholder="Daily Rate" min="12" max="1000" maxLength="4"/></div>
              </div>
              <button type="submit" className="btn btn-primary" onClick={this.updateBoarder}>Create Dog Boarding Profile</button>
          </div>

          <div id="parent-profile" className="collapse">
            <div className="mb-3"><img className="profile-pic" id="database-parent-pic" src={GenericPic} alt="Profile"/></div>
            <input type="file" id="database-parent-input" onChange={this.previewPic}/>
              <div className="col">
                <div className="my-4 row"><input className="profile-input" id="database-parent-name" type="text" placeholder="Name" maxLength="50"/></div>
                <div className="my-4 row"><input className="profile-input" id="database-parent-phone" type="text" placeholder="Phone Number" min="0" max="9999999999" maxLength="11"/></div>
                <div className="my-4 row"><input className="profile-input" id="database-parent-city" type="text" placeholder="City" maxLength="50"/></div>
                {/* <button className="my-1 btn" id="parent-expand" onClick={this.expand}>Show More ▼</button>
                <div className="collapse" id="parent-featured-images">
                  <b>Featured Images</b>
                  <div><input className="feature-input" type="file" id="database-parent-feature1-input" onChange={this.previewPic}/></div>
                  <div><input className="feature-input" type="file" id="database-parent-feature2-input" onChange={this.previewPic}/></div>
                  <div><input className="feature-input" type="file" id="database-parent-feature3-input" onChange={this.previewPic}/></div>
                  <div><input className="feature-input" type="file" id="database-parent-feature4-input" onChange={this.previewPic}/></div>
                  <img className="feature-pic" id="database-parent-feature1-pic"/>
                  <img className="feature-pic" id="database-parent-feature2-pic"/>
                  <img className="feature-pic" id="database-parent-feature3-pic"/>
                  <img className="feature-pic" id="database-parent-feature4-pic"/>
                </div>  */}
              </div>
              <button type="submit" className="btn btn-primary" onClick={this.updateParent}>Edit Profile</button>
          </div>

          <div id="walker-profile" className="collapse">
            <div className="mb-3"><img className="profile-pic" id="database-walker-pic" src={GenericPic} alt="Profile"/></div>
            <input type="file" id="database-walker-input" onChange={this.previewPic}/>
              <div className="col">
                <div className="my-4 row"><input className="profile-input" id="database-walker-name" type="text" placeholder="Name" maxLength="50"/></div>
                <div className="my-4 row"><input className="profile-input" id="database-walker-phone" type="text" placeholder="Phone Number" min="0" max="9999999999" maxLength="11"/></div>
                <div className="my-4 row"><input className="profile-input" id="database-walker-city" type="text" placeholder="City" maxLength="50"/></div>
                <div className="my-4 row"><input className="profile-input" id="database-walker-hourly-rate" type="text" placeholder="Hourly Rate" min="12" max="100" maxLength="3"/></div>
                <button className="my-1 btn" id="walker-expand" onClick={this.expand}>Show More ▼</button>
                <div className="collapse" id="walker-featured-images">
                  <b>Featured Images</b>
                  <div><input className="feature-input" type="file" id="database-walker-feature1-input" onChange={this.previewPic}/></div>
                  <div><input className="feature-input" type="file" id="database-walker-feature2-input" onChange={this.previewPic}/></div>
                  <div><input className="feature-input" type="file" id="database-walker-feature3-input" onChange={this.previewPic}/></div>
                  <div><input className="feature-input" type="file" id="database-walker-feature4-input" onChange={this.previewPic}/></div>
                  <img className="feature-pic" id="database-walker-feature1-pic"/>
                  <img className="feature-pic" id="database-walker-feature2-pic"/>
                  <img className="feature-pic" id="database-walker-feature3-pic"/>
                  <img className="feature-pic" id="database-walker-feature4-pic"/>
                </div>
              </div>
              <button type="submit" className="btn btn-primary" onClick={this.updateWalker}>Edit Profile</button>
          </div>

          <div id="boarder-profile" className="collapse">
            <div className="mb-3"><img className="profile-pic" id="database-boarder-pic" src={GenericPic} alt="Profile"/></div>
            <input type="file" id="database-boarder-input" onChange={this.previewPic}/>
            <div className="row">
              <div className="col">
                <div className="my-4 row"><input className="profile-input" id="database-boarder-name" type="text" placeholder="Name" maxLength="50"/></div>
                <div className="my-4 row"><input className="profile-input" id="database-boarder-phone" type="text" placeholder="Phone Number" min="0" max="9999999999" maxLength="11"/></div>
                <div className="my-4 row"><input className="profile-input" id="database-boarder-address" type="text" placeholder="Address" maxLength="50"/></div>
                <div className="my-4 row"><input className="profile-input" id="database-boarder-city" type="text" placeholder="City" maxLength="50"/></div>
                <div className="mt-4 row"><input className="profile-input" id="database-boarder-daily-rate" type="text" placeholder="Daily Rate" min="12" max="1000" maxLength="4"/></div>
                <button className="my-1 btn" id="boarder-expand" onClick={this.expand}>Show More ▼</button>
                <div className="collapse" id="boarder-featured-images">
                  <b>Featured Images</b>
                  <div><input className="feature-input" type="file" id="database-boarder-feature1-input" onChange={this.previewPic}/></div>
                  <div><input className="feature-input" type="file" id="database-boarder-feature2-input" onChange={this.previewPic}/></div>
                  <div><input className="feature-input" type="file" id="database-boarder-feature3-input" onChange={this.previewPic}/></div>
                  <div><input className="feature-input" type="file" id="database-boarder-feature4-input" onChange={this.previewPic}/></div>
                  <img className="feature-pic" id="database-boarder-feature1-pic"/>
                  <img className="feature-pic" id="database-boarder-feature2-pic"/>
                  <img className="feature-pic" id="database-boarder-feature3-pic"/>
                  <img className="feature-pic" id="database-boarder-feature4-pic"/>
                </div>
              </div>
            </div>
            <button type="submit" className="btn btn-primary mt-4" onClick={this.updateBoarder}>Edit Profile</button>
          </div>

          <div id="upload-loader" className="mt-4"><Loader type="Oval" color="black" height={75} width={75}/></div>
          
          <div id="delete-popup" className="fixed-top collapse">
              <div className="pooch-navbar-item">Are you sure you want to delete your profile? This will allow you to change account type</div>
              <div><button className="my-2 mr-4 btn-danger popup-button" onClick={this.deleteProfile}>Yes</button><button className="my-2 ml-4 btn-primary popup-button" onClick={this.closeDeletePopup}>No</button></div>
              <div className="trak_body">WARNING: This is irreversable!</div>
          </div>
          <div id="delete-button-row" className="collapse"><button type="submit" className="btn btn-danger mt-8" onClick={this.openDeletePopup}>Delete Profile</button></div>
          <div className="fixed-top" id="error-message">
            <div>Image size too large</div>
            <button className="my-2 mr-4 btn-danger popup-button" onClick={this.closeError}>Close</button>
          </div>
          </Fade>
      </div>
    ); 
  }
};

export default ProfileForm;
