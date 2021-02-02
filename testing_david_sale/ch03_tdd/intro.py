# Test driven development is often a key part of the agile development process, which advocates
# iterative development over more restrictive processes such as waterfall. TDD in agile
# makes testing your focus up front, rather than an afterthought at the end of development

# big
# plus point to this process is that it removes the pressure of deadlines to some degree and
# teams can work to deliver a minimum viable product within a certain time frame. Basically
# this means you deliver the smallest product that you could release to customers that would
# add business value. Once you have developed this product, you can then continue to iterate

# The agile manifesto was drafted in February 2001. It aimed to offer guidance and explain the
# fundamentals of the agile development process. The manifesto resulted in the following
# 12 principles:
# ■ Increasing customer satisfaction by rapid delivery of useful software
# ■ Accommodating changing requirements, even late in development
# ■ Delivering working software frequently (weeks rather than months)
# ■ Using working software as the principal measure of progress
# ■ Building a sustainable development process and being able to maintain a constant pace
# ■ Providing close, daily cooperation between business people and developers
# ■ Communicating through face-to-face conversation, which is the best form of communication
# (co-location)
# ■ Building projects around motivated individuals who should be trusted
# ■ Paying continuous attention to technical excellence and good design
# ■ Exercising simplicity—the art of maximizing the amount of work not done
# ■ Creating self-organizing teams
# ■ Making regular adaptations to changing circumstances

# concepts. The average iteration in an agile team is known as a
# “sprint.

# 1. Sprint planning/planning games: Here the development team is presented with a
# series of stories (business requirements for the application being developed). Team members
# discuss how difficult the story is and what it will entail.

# 2. Development: Stories can then be picked up from the agile board, which is typically a
# whiteboard displayed somewhere near the development team and that is split into sections
# to show the progress of any one story card. A usual split for a board such as this
# would be Sprint Backlog, In Progress, QA/Test, and Done. A developer will pick up a
# card from the Backlog and move it to the In Progress column.

# When the developers
# are happy the card is complete, they move it to the QA/Test column, where the
# Quality Assurance (QA) personnel on the team can write more tests, perform manual
# tests, and possibly performance test the work to their satisfaction

# 3. Showcase: An important part of the process, as mentioned previously, is to deliver a
# “working” product at the end of each sprint. The term “working” is used loosely here, in
# that it does not mean complete or ready for the end user. Simply, it means that some
# functionality of the overall application is now complete or ready to be iterated on further.

# 4. Retrospective: The final pillar of the agile process is the retrospective, which takes
# place at the end of a sprint. The retrospective is some dedicated time for the team to
# reflect on the previous sprint and talk about things that went wrong, went well, or
# could have been different

# The basic concept of TDD is to write a failing test first, before writing any code. You may ask how
# can I write a test before I know what it does? Indeed, this is a valid question that will hopefully
# become clearer over the course of the chapter. Essentially, the tests should drive your development
# by failing in a way that allows you to write a piece of code. For example, if you need a class
# to represent some data, and a method to access that data, then a test could call this new method
# on the class as if it existed. Then your test would fail indicating that the method and class do not
# exist and you can begin developing them.

# What experienced developers often
# find is that if too many bugs are getting through to your released code in production, then
# you have not got enough coverage in your tests for all the scenarios your code faces  On the other hand, if it becomes very difficult to make even small changes to your application
# (because the changes cause many tests to fail, which require updating) then you probably
# have too many tests.

# With your tests in place and the code now making them pass, you can enter the next stage,
# where you have the opportunity to refactor your code. Because in this process, you write the
# minimum amount of code necessary to get the tests to pass, your code may not even be functional
# except for passing the test cases.

# Ping-Pong Programming
# Ping-pong programming can make pair programming more fun and engaging for the developers
# involved. With pair programming, the idea is that one person is writing the code while
# the other is checking the code being written. The pair talk with each other over the design of
# the code, discuss what they each think is required, and develop the functionality until
# requirements are met. One person writes the failing test, and the other writes the code to make it
# pass.


# requirements ex:

# You could consider that the bank account class would have the following business
# requirements:
# ■ The customer must be able to uniquely access his bank account and retrieve the
# balance.
# ■ The customer must be able to deposit funds into her bank account.
# ■ The customer must be able to withdraw funds from his bank account.
# These requirements allow you to deliver a minimum viable product at the end of this sprint.
# If you deliver these requirements, then you have code that can handle the basic responsibilities
# of a bank account.


