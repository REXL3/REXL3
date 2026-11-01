## New features and improvements

- Production accepts an owner vision exception in place of a proven does-not-fit justification for a
  model that ships with vision off. The exception is a record in each vision-off profile's sealed
  qualification receipt: the maintainer who granted it (am423 or vcruz305), the grant date, an expiry
  date and the reason. The model-branch completeness check accepts vision off when every profile
  carries a valid one. (PR #53)

## Breaking changes and upgrade notes

- The verifier now refuses a vision-off qualification receipt with no owner vision exception, or with
  an expired or malformed one, and refuses an expired or malformed record on a vision-on receipt. The
  expiry is checked against the date of each verification run. (PR #53)
