import React from 'react'

const dog_wash = ({ dog_wash }) => {
    return (
      <div>
        {dog_wash.map((dog_wash) => (
          <div class="dogwash">
            <div class="dog-body">
              <h5 class="dog-title">{dog_wash.name}</h5>
              <h6 class="dog-address">{dog_wash.address}</h6>

            </div>
          </div>
        ))}
      </div>
    )
  };

  export default dog_wash